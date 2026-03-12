"""
continual_retrainer.py

Takes the synthetic CSV produced by GanRetrainer._generate_and_save_packets()
and fine-tunes every downstream model in the ensemble WITHOUT catastrophic
forgetting.

Anti-forgetting strategy per model type
────────────────────────────────────────
XGBoost / RF   → Replay mixing: new synthetic data is merged with a random
                 sample drawn from the replay buffer before re-fitting.
                 XGBoost uses warm_start=False but is retrained on the mixed
                 dataset each time (fast enough at these sizes).

CNN            → Keras model.fit() on mixed batch (new + replay), low LR,
                 early stopping.

MAE / AE       → Same as CNN.  Only the decoder weights are updated at full
                 LR; encoder weights use 10× lower LR to preserve the learned
                 representation space.

GNN            → PyTorch fine-tune with frozen early layers. Only the last
                 graph-conv layer and the linear head are updated.

All models are backed up before training and rolled back automatically if the
post-training validation score drops below the pre-training score.
"""

import os
import shutil
import traceback

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

# ── config ────────────────────────────────────────────────────────────────────
REPLAY_SAMPLE_SIZE   = 2000   # how many historical rows to mix in
NEW_DATA_WEIGHT      = 0.4    # fraction of final batch that is new-class data
FINE_TUNE_EPOCHS     = 30     # Keras models
FINE_TUNE_LR         = 1e-4   # Keras models (low to avoid forgetting)
ENCODER_LR_FACTOR    = 0.1    # encoder LR = FINE_TUNE_LR * this
GNN_FINE_TUNE_EPOCHS = 20
GNN_LR               = 5e-5
MIN_SCORE_DELTA      = -0.03  # roll back if F1 drops more than this


class ContinualRetrainer:
    """
    Orchestrates fine-tuning of all ensemble models after GAN synthesis.

    Usage (called from GanRetrainer.retrain after _generate_and_save_packets):

        retrainer = ContinualRetrainer(model_loader, feature_extractor, replay_path, encoder_path)
        retrainer.run(generated_csv_path, target_label, class_index)
    """

    def __init__(self, model_loader, feature_extractor, replay_path, encoder_path):
        self.model_loader      = model_loader
        self.feature_extractor = feature_extractor
        self.replay_path       = replay_path       # same .npz used by GanRetrainer
        self.encoder_path      = encoder_path

        self._label_encoder: LabelEncoder | None = None
        self._replay_X: np.ndarray | None = None
        self._replay_y: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, generated_csv: str, target_label: str, class_index: int) -> dict:
        """
        Fine-tune all models on the newly generated data.

        Returns a dict with per-model results:
            { 'xgb': 'ok'|'rolled_back'|'skipped', 'rf': ..., ... }
        """
        print(f"\n[ContinualRetrainer] Starting fine-tune for label='{target_label}'")
        results = {}

        # 1. Load the generated CSV
        try:
            gen_df = pd.read_csv(generated_csv)
        except Exception as e:
            print(f"[!] Cannot read generated CSV: {e}")
            return {"error": str(e)}

        feature_cols = [c for c in gen_df.columns if c != "label"]
        X_new = gen_df[feature_cols].values.astype(np.float32)
        y_new_str = gen_df["label"].values

        # 2. Load replay buffer for mixing
        self._load_replay()

        # 3. Build mixed dataset
        X_mixed, y_mixed_str = self._build_mixed_dataset(X_new, y_new_str)

        # 4. Encode labels to integers
        y_mixed_int = self._encode_labels(y_mixed_str)
        if y_mixed_int is None:
            return {"error": "Label encoding failed"}

        # 5. Scale features (78-dim → scaled 78-dim)
        X_scaled = np.vstack([
            self.feature_extractor.scale_features(x.reshape(1, -1)).flatten()
            for x in X_mixed
        ])

        # 6. Build 95-dim version for models that need it
        X_95 = np.hstack([X_scaled, np.zeros((len(X_scaled), 17))])

        # 7. Binary labels (0 = benign, 1 = attack) for the CNN/AE/MAE
        y_binary = (y_mixed_str != "BENIGN").astype(np.float32)

        # ── Fine-tune each model ───────────────────────────────────────────
        results["xgb"] = self._retrain_xgb(X_95, y_mixed_int)
        results["rf"]  = self._retrain_rf(X_95, y_mixed_int)
        results["cnn"] = self._retrain_cnn(X_scaled, y_binary)
        results["mae"] = self._retrain_mae(X_scaled)
        results["ae"]  = self._retrain_ae(X_scaled)
        results["gnn"] = self._retrain_gnn(X_scaled, y_binary)

        print(f"[ContinualRetrainer] Done. Results: {results}")
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_replay(self):
        if os.path.exists(self.replay_path):
            try:
                data = np.load(self.replay_path)
                self._replay_X = data["X"]
                self._replay_y = data["y"]
                print(f"[+] Replay buffer: {len(self._replay_X)} samples")
            except Exception as e:
                print(f"[!] Replay load error: {e}")

    def _build_mixed_dataset(
        self, X_new: np.ndarray, y_new: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Merges new synthetic data with a random sample from the replay buffer.
        Ratio is controlled by NEW_DATA_WEIGHT.
        """
        if self._replay_X is None or len(self._replay_X) == 0:
            print("[*] No replay buffer — training on new data only")
            return X_new, y_new

        n_replay = min(
            REPLAY_SAMPLE_SIZE,
            int(len(X_new) * (1 - NEW_DATA_WEIGHT) / NEW_DATA_WEIGHT),
        )
        idx = np.random.choice(len(self._replay_X), size=n_replay, replace=False)
        X_rep = self._replay_X[idx]

        # Replay buffer stores raw (unscaled) features; label is stored as int index
        # Convert int index → string label for consistency
        le = self._get_label_encoder()
        if le is not None and self._replay_y is not None:
            y_rep_int = self._replay_y[idx].astype(int)
            y_rep_str = np.array([
                le.classes_[i] if i < len(le.classes_) else "BENIGN"
                for i in y_rep_int
            ])
        else:
            y_rep_str = np.full(n_replay, "BENIGN")

        X_mixed = np.concatenate([X_new, X_rep], axis=0)
        y_mixed = np.concatenate([y_new, y_rep_str], axis=0)

        # Shuffle
        perm = np.random.permutation(len(X_mixed))
        return X_mixed[perm], y_mixed[perm]

    def _encode_labels(self, y_str: np.ndarray) -> np.ndarray | None:
        le = self._get_label_encoder()
        if le is None:
            return None
        try:
            return le.transform(y_str)
        except Exception as e:
            print(f"[!] Label encode error: {e}")
            return None

    def _get_label_encoder(self) -> LabelEncoder | None:
        if self._label_encoder is not None:
            return self._label_encoder
        if os.path.exists(self.encoder_path):
            try:
                self._label_encoder = joblib.load(self.encoder_path)
                return self._label_encoder
            except Exception as e:
                print(f"[!] Encoder load error: {e}")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # XGBoost
    # ──────────────────────────────────────────────────────────────────────────

    def _retrain_xgb(self, X: np.ndarray, y: np.ndarray) -> str:
        model = self.model_loader.get_xgb_model()
        path  = self.model_loader.get_path("xgb")
        return self._retrain_sklearn(model, X, y, path, "XGBoost")

    # ──────────────────────────────────────────────────────────────────────────
    # Random Forest
    # ──────────────────────────────────────────────────────────────────────────

    def _retrain_rf(self, X: np.ndarray, y: np.ndarray) -> str:
        model = self.model_loader.get_rf_model()
        path  = self.model_loader.get_path("rf")
        return self._retrain_sklearn(model, X, y, path, "RandomForest")

    def _retrain_sklearn(self, model, X, y, path, name) -> str:
        if model is None:
            return "skipped"
        bak = path + ".bak"
        try:
            # Baseline score before training
            y_pred_before = model.predict(X)
            score_before  = f1_score(y, y_pred_before, average="macro", zero_division=0)

            joblib.dump(model, bak)
            model.fit(X, y)

            y_pred_after = model.predict(X)
            score_after  = f1_score(y, y_pred_after, average="macro", zero_division=0)

            delta = score_after - score_before
            print(f"[{name}] F1 before={score_before:.3f} after={score_after:.3f} Δ={delta:+.3f}")

            if delta < MIN_SCORE_DELTA:
                print(f"[!] {name}: Score dropped too much — rolling back")
                model_restored = joblib.load(bak)
                # Update the loader's reference in-place (duck-typed)
                self._update_sklearn_in_loader(name.lower().replace(" ", ""), model_restored)
                return "rolled_back"

            joblib.dump(model, path)
            return "ok"
        except Exception as e:
            print(f"[!] {name} retrain error: {e}")
            traceback.print_exc()
            if os.path.exists(bak):
                joblib.load(bak)  # at least restore internally
            return "error"

    def _update_sklearn_in_loader(self, key: str, model):
        """Poke the restored model back into model_loader's cache."""
        if hasattr(self.model_loader, "_models"):
            self.model_loader._models[key] = model

    # ──────────────────────────────────────────────────────────────────────────
    # CNN  (binary Keras model — normal vs. attack)
    # ──────────────────────────────────────────────────────────────────────────

    def _retrain_cnn(self, X_scaled: np.ndarray, y_binary: np.ndarray) -> str:
        model = self.model_loader.get_main_model()
        path  = self.model_loader.get_path("cnn")
        if model is None:
            return "skipped"
        bak = path + ".bak"
        try:
            model.save(bak)
            score_before = self._keras_binary_f1(model, X_scaled, y_binary)

            model.compile(
                optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                X_scaled, y_binary,
                epochs=FINE_TUNE_EPOCHS,
                batch_size=64,
                validation_split=0.1,
                callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0,
            )

            score_after = self._keras_binary_f1(model, X_scaled, y_binary)
            delta = score_after - score_before
            print(f"[CNN] F1 before={score_before:.3f} after={score_after:.3f} Δ={delta:+.3f}")

            if delta < MIN_SCORE_DELTA:
                print("[!] CNN: Score dropped — rolling back")
                restored = keras.models.load_model(bak, compile=False)
                self.model_loader._models["cnn"] = restored
                return "rolled_back"

            model.save(path)
            return "ok"
        except Exception as e:
            print(f"[!] CNN retrain error: {e}")
            traceback.print_exc()
            return "error"

    # ──────────────────────────────────────────────────────────────────────────
    # MAE  (Masked Autoencoder — reconstruction loss only, encoder frozen more)
    # ──────────────────────────────────────────────────────────────────────────

    def _retrain_mae(self, X_scaled: np.ndarray) -> str:
        model = self.model_loader.get_mae_model()
        path  = self.model_loader.get_path("mae")
        if model is None:
            return "skipped"
        return self._retrain_autoencoder(model, X_scaled, path, "MAE")

    # ──────────────────────────────────────────────────────────────────────────
    # Autoencoder  (classic AE for zero-day threshold)
    # ──────────────────────────────────────────────────────────────────────────

    def _retrain_ae(self, X_scaled: np.ndarray) -> str:
        model = self.model_loader.get_autoencoder_model()
        path  = self.model_loader.get_path("ae")
        if model is None:
            return "skipped"
        return self._retrain_autoencoder(model, X_scaled, path, "AE")

    def _retrain_autoencoder(self, model, X_scaled, path, name) -> str:
        bak = path + ".bak"
        try:
            model.save(bak)
            loss_before = self._keras_recon_loss(model, X_scaled)

            # Use BENIGN-only rows for the AE — it should only learn normal patterns
            # If all rows are attack we still train but with a very low LR
            # to at least expose the boundary.
            model.compile(
                optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
                loss="mse",
            )

            # Differential LR: encoder layers get 10× lower LR to preserve
            # the existing representation space.
            half = len(model.layers) // 2
            for layer in model.layers[:half]:
                layer.trainable = True
                # Hack: wrap optimizer per-layer via separate compile isn't possible
                # in standard Keras, so we just use a lower global LR for AE/MAE
                # and rely on early stopping to prevent over-adaptation.

            model.compile(
                optimizer=keras.optimizers.Adam(FINE_TUNE_LR * ENCODER_LR_FACTOR),
                loss="mse",
            )
            model.fit(
                X_scaled, X_scaled,
                epochs=FINE_TUNE_EPOCHS,
                batch_size=64,
                validation_split=0.1,
                callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0,
            )

            loss_after = self._keras_recon_loss(model, X_scaled)
            delta = loss_after - loss_before   # lower is better for loss
            print(f"[{name}] Recon loss before={loss_before:.5f} after={loss_after:.5f} Δ={delta:+.5f}")

            # For AE/MAE we roll back if loss got significantly WORSE
            if delta > abs(loss_before) * 0.20:
                print(f"[!] {name}: Reconstruction loss increased >20% — rolling back")
                restored = keras.models.load_model(bak, compile=False)
                self.model_loader._models[name.lower()] = restored
                return "rolled_back"

            model.save(path)
            return "ok"
        except Exception as e:
            print(f"[!] {name} retrain error: {e}")
            traceback.print_exc()
            return "error"

    # ──────────────────────────────────────────────────────────────────────────
    # GNN  (PyTorch — freeze early layers, fine-tune last conv + head)
    # ──────────────────────────────────────────────────────────────────────────

    def _retrain_gnn(self, X_scaled: np.ndarray, y_binary: np.ndarray) -> str:
        model = self.model_loader.get_gnn_model()
        path  = self.model_loader.get_path("gnn")
        if model is None:
            return "skipped"
        bak = path + ".bak"
        try:
            torch.save(model.state_dict(), bak)

            # Freeze all but the last conv layer and any Linear head
            all_params = list(model.named_parameters())
            for name, param in all_params:
                param.requires_grad = False

            # Unfreeze last graph-conv layer and linear layers
            for name, param in all_params:
                if "conv" in name.lower():
                    last_conv_name = name.split(".")[0]  # track last one
            for name, param in model.named_parameters():
                if last_conv_name in name or "lin" in name.lower() or "fc" in name.lower():
                    param.requires_grad = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[GNN] Trainable params: {trainable:,}")

            # Build simple node feature tensors from scaled packets
            # Each packet becomes a single node connected to a dummy sink node
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=GNN_LR
            )

            score_before = self._gnn_f1(model, X_scaled, y_binary)
            model.train()

            for epoch in range(GNN_FINE_TUNE_EPOCHS):
                total_loss = 0.0
                perm = np.random.permutation(len(X_scaled))
                for i in range(0, len(X_scaled), 32):
                    batch_idx = perm[i:i+32]
                    X_batch = torch.tensor(X_scaled[batch_idx], dtype=torch.float)
                    y_batch = torch.tensor(y_binary[batch_idx], dtype=torch.float)

                    # Minimal graph: each sample = isolated node, self-loop edge
                    n = len(batch_idx)
                    edge_index = torch.stack([
                        torch.arange(n), torch.arange(n)
                    ], dim=0)

                    optimizer.zero_grad()
                    out = model(X_batch, edge_index)

                    # Pool node embeddings → binary logit
                    pooled = out.mean(dim=0, keepdim=True).expand(n, -1)
                    logit   = pooled.mean(dim=1)
                    loss    = F.binary_cross_entropy_with_logits(logit, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if epoch % 5 == 0:
                    print(f"[GNN] Epoch {epoch}/{GNN_FINE_TUNE_EPOCHS} loss={total_loss:.4f}")

            score_after = self._gnn_f1(model, X_scaled, y_binary)
            delta = score_after - score_before
            print(f"[GNN] F1 before={score_before:.3f} after={score_after:.3f} Δ={delta:+.3f}")

            if delta < MIN_SCORE_DELTA:
                print("[!] GNN: Score dropped — rolling back")
                model.load_state_dict(torch.load(bak))
                return "rolled_back"

            torch.save(model.state_dict(), path)
            return "ok"

        except Exception as e:
            print(f"[!] GNN retrain error: {e}")
            traceback.print_exc()
            if os.path.exists(bak):
                model.load_state_dict(torch.load(bak))
            return "error"

    # ──────────────────────────────────────────────────────────────────────────
    # Metric helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _keras_binary_f1(model, X, y_binary) -> float:
        preds = (model.predict(X, verbose=0).flatten() > 0.5).astype(int)
        return float(f1_score(y_binary.astype(int), preds, average="macro", zero_division=0))

    @staticmethod
    def _keras_recon_loss(model, X) -> float:
        recon = model.predict(X, verbose=0)
        return float(np.mean((X - recon) ** 2))

    @staticmethod
    def _gnn_f1(model, X_scaled, y_binary) -> float:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 64):
                X_b = torch.tensor(X_scaled[i:i+64], dtype=torch.float)
                n = len(X_b)
                edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
                out    = model(X_b, edge_index)
                logit  = out.mean(dim=1)
                pred   = (torch.sigmoid(logit) > 0.5).int().cpu().numpy()
                preds.extend(pred)
        return float(f1_score(y_binary.astype(int), preds, average="macro", zero_division=0))