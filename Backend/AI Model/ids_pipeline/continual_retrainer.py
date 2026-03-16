"""
continual_retrainer.py

Fixes applied vs previous version:
  1. MAE  — uses torch.save/load_state_dict instead of Keras model.save()
  2. AE   — removed double compile; freezes encoder layers properly with
             layer.trainable = False before the single compile
  3. RF rollback — _update_sklearn_in_loader now maps "randomforest" → "rf"
                   so the loader cache is actually updated on rollback
"""

import os
import traceback

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

REPLAY_SAMPLE_SIZE   = 2000
NEW_DATA_WEIGHT      = 0.4
FINE_TUNE_EPOCHS     = 30
FINE_TUNE_LR         = 1e-4
ENCODER_LR_FACTOR    = 0.1    # AE encoder gets this fraction of FINE_TUNE_LR
GNN_FINE_TUNE_EPOCHS = 20
GNN_LR               = 5e-5
MIN_SCORE_DELTA      = -0.03


class ContinualRetrainer:

    def __init__(self, model_loader, feature_extractor, replay_path, encoder_path):
        self.model_loader      = model_loader
        self.feature_extractor = feature_extractor
        self.replay_path       = replay_path
        self.encoder_path      = encoder_path
        self._label_encoder    = None
        self._replay_X         = None
        self._replay_y         = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, generated_csv: str, target_label: str, class_index: int) -> dict:
        print(f"\n[ContinualRetrainer] Fine-tuning for label='{target_label}'")
        results = {}

        try:
            gen_df = pd.read_csv(generated_csv)
        except Exception as e:
            print(f"[!] Cannot read generated CSV: {e}")
            return {"error": str(e)}

        feature_cols = [c for c in gen_df.columns if c != "label"]
        X_new     = gen_df[feature_cols].values.astype(np.float32)
        y_new_str = gen_df["label"].values

        self._load_replay()
        X_mixed, y_mixed_str = self._build_mixed_dataset(X_new, y_new_str)

        y_mixed_int = self._encode_labels(y_mixed_str)
        if y_mixed_int is None:
            return {"error": "Label encoding failed"}

        X_scaled = np.vstack([
            self.feature_extractor.scale_features(x.reshape(1, -1)).flatten()
            for x in X_mixed
        ])
        X_95     = np.hstack([X_scaled, np.zeros((len(X_scaled), 17))])
        y_binary = (y_mixed_str != "BENIGN").astype(np.float32)

        results["xgb"] = self._retrain_xgb(X_95, y_mixed_int)
        results["rf"]  = self._retrain_rf(X_95, y_mixed_int)
        results["cnn"] = self._retrain_cnn(X_scaled, y_binary)
        results["ae"]  = self._retrain_ae(X_scaled)
        results["mae"] = self._retrain_mae(X_scaled)
        results["gnn"] = self._retrain_gnn(X_scaled, y_binary)

        print(f"[ContinualRetrainer] Done. {results}")
        return results

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _load_replay(self):
        if os.path.exists(self.replay_path):
            try:
                data           = np.load(self.replay_path)
                self._replay_X = data["X"]
                self._replay_y = data["y"]
                print(f"[+] Replay buffer: {len(self._replay_X)} samples")
            except Exception as e:
                print(f"[!] Replay load error: {e}")

    def _build_mixed_dataset(self, X_new, y_new):
        if self._replay_X is None or len(self._replay_X) == 0:
            print("[*] No replay buffer — new data only")
            return X_new, y_new

        n_replay = min(
            REPLAY_SAMPLE_SIZE,
            int(len(X_new) * (1 - NEW_DATA_WEIGHT) / NEW_DATA_WEIGHT),
        )
        idx   = np.random.choice(len(self._replay_X), size=n_replay, replace=False)
        X_rep = self._replay_X[idx]

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
        perm    = np.random.permutation(len(X_mixed))
        return X_mixed[perm], y_mixed[perm]

    def _encode_labels(self, y_str):
        le = self._get_label_encoder()
        if le is None:
            return None
        try:
            return le.transform(y_str)
        except Exception as e:
            print(f"[!] Label encode error: {e}")
            return None

    def _get_label_encoder(self):
        if self._label_encoder is not None:
            return self._label_encoder
        if os.path.exists(self.encoder_path):
            try:
                self._label_encoder = joblib.load(self.encoder_path)
                return self._label_encoder
            except Exception as e:
                print(f"[!] Encoder load error: {e}")
        return None

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    def _retrain_xgb(self, X, y):
        return self._retrain_sklearn(
            self.model_loader.get_xgb_model(),
            X, y,
            self.model_loader.get_path("xgb"),
            "XGBoost", "xgb",
        )

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------

    def _retrain_rf(self, X, y):
        return self._retrain_sklearn(
            self.model_loader.get_rf_model(),
            X, y,
            self.model_loader.get_path("rf"),
            "RandomForest", "rf",          # FIX: pass explicit loader key "rf"
        )

    def _retrain_sklearn(self, model, X, y, path, name, loader_key):
        if model is None:
            return "skipped"
        bak = path + ".bak"
        try:
            y_pred_before = model.predict(X)
            score_before  = f1_score(y, y_pred_before, average="macro", zero_division=0)

            joblib.dump(model, bak)
            model.fit(X, y)

            y_pred_after = model.predict(X)
            score_after  = f1_score(y, y_pred_after, average="macro", zero_division=0)
            delta = score_after - score_before
            print(f"[{name}] F1 {score_before:.3f} → {score_after:.3f} (Δ{delta:+.3f})")

            if delta < MIN_SCORE_DELTA:
                print(f"[!] {name}: rolling back")
                restored = joblib.load(bak)
                # FIX: use the explicit loader_key, not a derived string
                if hasattr(self.model_loader, "_models"):
                    self.model_loader._models[loader_key] = restored
                return "rolled_back"

            joblib.dump(model, path)
            return "ok"
        except Exception as e:
            print(f"[!] {name} error: {e}")
            traceback.print_exc()
            return "error"

    # ------------------------------------------------------------------
    # CNN  (Keras binary classifier)
    # ------------------------------------------------------------------

    def _retrain_cnn(self, X_scaled, y_binary):
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
                loss="binary_crossentropy", metrics=["accuracy"],
            )
            model.fit(
                X_scaled, y_binary,
                epochs=FINE_TUNE_EPOCHS, batch_size=64, validation_split=0.1,
                callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0,
            )

            score_after = self._keras_binary_f1(model, X_scaled, y_binary)
            delta = score_after - score_before
            print(f"[CNN] F1 {score_before:.3f} → {score_after:.3f} (Δ{delta:+.3f})")

            if delta < MIN_SCORE_DELTA:
                print("[!] CNN: rolling back")
                restored = keras.models.load_model(bak, compile=False)
                self.model_loader._models["cnn"] = restored
                return "rolled_back"

            model.save(path)
            return "ok"
        except Exception as e:
            print(f"[!] CNN error: {e}")
            traceback.print_exc()
            return "error"

    # ------------------------------------------------------------------
    # AE  (Keras autoencoder)
    #
    # FIX: removed double-compile; encoder half is frozen with
    #      layer.trainable = False before the single compile call.
    # ------------------------------------------------------------------

    def _retrain_ae(self, X_scaled):
        model = self.model_loader.get_autoencoder_model()
        path  = self.model_loader.get_path("ae")
        if model is None:
            return "skipped"
        bak = path + ".bak"
        try:
            model.save(bak)
            loss_before = self._keras_recon_loss(model, X_scaled)

            # Freeze the encoder half (first 50% of layers) to preserve the
            # learned representation space. Only the decoder adapts.
            half = len(model.layers) // 2
            for layer in model.layers[:half]:
                layer.trainable = False
            for layer in model.layers[half:]:
                layer.trainable = True

            # Single compile — low LR because only decoder is updating
            model.compile(
                optimizer=keras.optimizers.Adam(FINE_TUNE_LR * ENCODER_LR_FACTOR),
                loss="mse",
            )
            model.fit(
                X_scaled, X_scaled,
                epochs=FINE_TUNE_EPOCHS, batch_size=64, validation_split=0.1,
                callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0,
            )

            # Unfreeze for future training runs
            for layer in model.layers:
                layer.trainable = True

            loss_after = self._keras_recon_loss(model, X_scaled)
            delta = loss_after - loss_before
            print(f"[AE] Recon loss {loss_before:.5f} → {loss_after:.5f} (Δ{delta:+.5f})")

            if delta > abs(loss_before) * 0.20:
                print("[!] AE: loss increased >20% — rolling back")
                restored = keras.models.load_model(bak, compile=False)
                self.model_loader._models["ae"] = restored
                return "rolled_back"

            model.save(path)
            return "ok"
        except Exception as e:
            print(f"[!] AE error: {e}")
            traceback.print_exc()
            return "error"

    # ------------------------------------------------------------------
    # MAE  (PyTorch masked autoencoder)
    #
    # FIX: uses torch.save/load_state_dict — NOT Keras model.save()
    # ------------------------------------------------------------------

    def _retrain_mae(self, X_scaled):
        model = self.model_loader.get_mae_model()
        path  = self.model_loader.get_path("mae")
        if model is None:
            return "skipped"
        bak = path + ".bak"
        try:
            # Save via PyTorch
            torch.save(model.state_dict(), bak)
            loss_before = self._torch_recon_loss(model, X_scaled)

            # Freeze encoder: freeze all parameters, then unfreeze decoder layers
            # MAE typically names decoder layers with "decoder" in the param name
            for name, param in model.named_parameters():
                param.requires_grad = "decoder" in name.lower()

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=FINE_TUNE_LR * ENCODER_LR_FACTOR,
            )

            model.train()
            dataset = torch.tensor(X_scaled, dtype=torch.float)
            for epoch in range(FINE_TUNE_EPOCHS):
                perm = torch.randperm(len(dataset))
                for i in range(0, len(dataset), 64):
                    batch = dataset[perm[i:i+64]]
                    optimizer.zero_grad()
                    recon, original = model(batch, mask_ratio=0.15)
                    loss = F.mse_loss(recon, original)
                    loss.backward()
                    optimizer.step()

            # Re-enable all params for future inference
            for param in model.parameters():
                param.requires_grad = True
            model.eval()

            loss_after = self._torch_recon_loss(model, X_scaled)
            delta = loss_after - loss_before
            print(f"[MAE] Recon loss {loss_before:.5f} → {loss_after:.5f} (Δ{delta:+.5f})")

            if delta > abs(loss_before) * 0.20:
                print("[!] MAE: loss increased >20% — rolling back")
                model.load_state_dict(torch.load(bak))
                model.eval()
                # load_state_dict modifies in place — loader reference stays valid
                return "rolled_back"

            torch.save(model.state_dict(), path)
            return "ok"
        except Exception as e:
            print(f"[!] MAE error: {e}")
            traceback.print_exc()
            if os.path.exists(bak):
                model.load_state_dict(torch.load(bak))
                model.eval()
            return "error"

    # ------------------------------------------------------------------
    # GNN  (PyTorch — freeze early layers, fine-tune last conv + head)
    # ------------------------------------------------------------------

    def _retrain_gnn(self, X_scaled, y_binary):
        model = self.model_loader.get_gnn_model()
        path  = self.model_loader.get_path("gnn")
        if model is None:
            return "skipped"
        bak = path + ".bak"
        try:
            torch.save(model.state_dict(), bak)

            # Find the name prefix of the last conv layer
            last_conv_name = None
            for name, _ in model.named_parameters():
                if "conv" in name.lower():
                    last_conv_name = name.split(".")[0]

            for name, param in model.named_parameters():
                param.requires_grad = (
                    last_conv_name is not None and last_conv_name in name
                ) or "lin" in name.lower() or "fc" in name.lower()

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[GNN] Trainable params: {trainable:,}")

            optimizer    = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=GNN_LR
            )
            score_before = self._gnn_f1(model, X_scaled, y_binary)
            model.train()

            for epoch in range(GNN_FINE_TUNE_EPOCHS):
                total_loss = 0.0
                perm = np.random.permutation(len(X_scaled))
                for i in range(0, len(X_scaled), 32):
                    idx     = perm[i:i+32]
                    X_batch = torch.tensor(X_scaled[idx], dtype=torch.float)
                    y_batch = torch.tensor(y_binary[idx], dtype=torch.float)
                    n       = len(idx)
                    edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
                    optimizer.zero_grad()
                    out    = model(X_batch, edge_index)
                    logit  = out.mean(dim=1)
                    loss   = F.binary_cross_entropy_with_logits(logit, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 5 == 0:
                    print(f"[GNN] Epoch {epoch}/{GNN_FINE_TUNE_EPOCHS} loss={total_loss:.4f}")

            for param in model.parameters():
                param.requires_grad = True
            model.eval()

            score_after = self._gnn_f1(model, X_scaled, y_binary)
            delta = score_after - score_before
            print(f"[GNN] F1 {score_before:.3f} → {score_after:.3f} (Δ{delta:+.3f})")

            if delta < MIN_SCORE_DELTA:
                print("[!] GNN: rolling back")
                model.load_state_dict(torch.load(bak))
                model.eval()
                return "rolled_back"

            torch.save(model.state_dict(), path)
            return "ok"
        except Exception as e:
            print(f"[!] GNN error: {e}")
            traceback.print_exc()
            if os.path.exists(bak):
                model.load_state_dict(torch.load(bak))
                model.eval()
            return "error"

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keras_binary_f1(model, X, y_binary):
        preds = (model.predict(X, verbose=0).flatten() > 0.5).astype(int)
        return float(f1_score(y_binary.astype(int), preds, average="macro", zero_division=0))

    @staticmethod
    def _keras_recon_loss(model, X):
        return float(np.mean((X - model.predict(X, verbose=0)) ** 2))

    @staticmethod
    def _torch_recon_loss(model, X_scaled):
        model.eval()
        with torch.no_grad():
            t     = torch.tensor(X_scaled, dtype=torch.float)
            recon, original = model(t, mask_ratio=0.15)
            return float(F.mse_loss(recon, original).item())

    @staticmethod
    def _gnn_f1(model, X_scaled, y_binary):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 64):
                X_b = torch.tensor(X_scaled[i:i+64], dtype=torch.float)
                n   = len(X_b)
                ei  = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
                out = model(X_b, ei)
                pred = (torch.sigmoid(out.mean(dim=1)) > 0.5).int().cpu().numpy()
                preds.extend(pred)
        return float(f1_score(y_binary.astype(int), preds, average="macro", zero_division=0))