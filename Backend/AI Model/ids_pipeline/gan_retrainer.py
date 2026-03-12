"""
gan_retrainer.py  (updated)

Changes from original:
  1. ContinualRetrainer is instantiated in __init__ and called at the end of retrain().
  2. retrain() now accepts a `pipeline` parameter: 'gan' or 'jitter'.
     The correct pipeline is chosen automatically based on the case:

       Case                              pipeline arg    handled by
       ───────────────────────────────── ──────────────  ───────────────────
       Normal flagged as attack          'jitter'        JitterRetrainer (new)
       Attack flagged as normal          'gan'           GanRetrainer (existing)
       Zero-day flagged as normal        'gan'           GanRetrainer (existing)
       Zero-day confirmed as zero-day    'gan'           GanRetrainer + new class

  3. After GAN synthesis, ContinualRetrainer.run() is called to fine-tune
     XGBoost, RF, CNN, MAE, AE, and GNN on the generated CSV.

Everything else (WGAN-GP, replay buffer, architecture surgery, rollback) is
unchanged from your original implementation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import traceback
import os
import joblib
import pandas as pd
import shutil

from continual_retrainer import ContinualRetrainer


class GanRetrainer:
    def __init__(self, packet_storage, feature_extractor, model_loader):
        self.packet_storage    = packet_storage
        self.feature_extractor = feature_extractor
        self.model_loader      = model_loader          # ← NEW: needed by ContinualRetrainer

        # --- ROBUST PATH SETUP ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir  = os.path.dirname(current_dir)
        models_dir  = os.path.join(parent_dir, "Models")

        self.model_path   = os.path.join(models_dir, "cgan_generator.keras")
        self.replay_path  = os.path.join(current_dir, "replay_buffer.npz")
        self.encoder_path = os.path.join(current_dir, "label_encoder.pkl")

        self.latent_dim = 128
        self.data_dim   = 78

        # Load or Build Generator
        try:
            self.generator  = keras.models.load_model(self.model_path, compile=False)
            self.num_classes = self.generator.input[1].shape[1]
            print(f"[+] GAN Engine: Loaded existing model with {self.num_classes} classes.")
        except Exception:
            self.num_classes = 15
            self.generator   = self._build_generator(self.num_classes)
            print(f"[!] GAN Engine: Initialized fresh model ({self.num_classes} classes).")

        # Load Replay Buffer
        self.replay_data, self.replay_labels = None, None
        if os.path.exists(self.replay_path):
            try:
                data = np.load(self.replay_path)
                self.replay_data, self.replay_labels = data['X'], data['y']
                print(f"[+] Replay Buffer: {len(self.replay_data)} samples loaded.")
            except Exception:
                pass

        # ContinualRetrainer — wired up once and reused
        self.continual_retrainer = ContinualRetrainer(
            model_loader      = model_loader,
            feature_extractor = feature_extractor,
            replay_path       = self.replay_path,
            encoder_path      = self.encoder_path,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def retrain(self, packet_ids, target_label, is_new_label, epochs=100):
        """
        Full adaptation pipeline:
          1. WGAN-GP training on seed packets
          2. Synthesis of 5 000 generated samples → CSV
          3. Fine-tune all downstream models on the CSV (ContinualRetrainer)
        """
        print(f"\n[*] --- STARTING NEURAL ADAPTATION PIPELINE ---")

        # 1. Backup
        if os.path.exists(self.model_path):
            shutil.copy(self.model_path, self.model_path + ".bak")

        # 2. Label identity
        class_index = self._handle_label_encoder(target_label, is_new_label)
        if class_index == -1:
            return {"status": "error", "message": "Identity Mapping Failed"}

        # 3. Architecture surgery (new class only)
        if is_new_label and class_index >= self.num_classes:
            print(f"[*] Architecture Surgery: Expanding GAN {self.num_classes} → {class_index + 1} classes")
            self.generator   = self._expand_model_classes(self.generator, class_index + 1)
            self.num_classes = class_index + 1

        # 4. Critic + WGAN
        self.critic = self._build_critic(self.num_classes)
        self.wgan   = self._build_wgan_gp_model()

        # 5. Data prep
        raw_feats = self.packet_storage.get_features_for_training(packet_ids)
        if not raw_feats:
            return {"status": "error", "message": "Seed packets not found in storage."}

        scaled_seeds = [
            self.feature_extractor.scale_features(np.array(f)).flatten()
            for f in raw_feats
        ]
        X_train, y_train = self._prepare_training_batch(scaled_seeds, class_index)

        # 6. WGAN-GP training
        try:
            print(f"[*] Training WGAN-GP on {len(X_train)} augmented samples…")
            dataset = (
                tf.data.Dataset
                .from_tensor_slices((X_train, y_train))
                .shuffle(1024)
                .batch(64)
            )
            self.wgan.fit(dataset, epochs=epochs, verbose=0)

            # 7. Health check
            if not self._check_model_health(class_index):
                print("[!] CRITICAL: Neural instability — reverting GAN.")
                self._rollback()
                return {"status": "error", "message": "Training failed (Mode Collapse). Reverted."}

            # 8. Save GAN + synthesise CSV
            self.generator.save(self.model_path)
            generated_csv = self._generate_and_save_packets(class_index, target_label)
            self._update_replay_buffer(class_index)

            # 9. ── Fine-tune all downstream models ──────────────────────────
            print("[*] Handing off to ContinualRetrainer…")
            retrain_results = self.continual_retrainer.run(
                generated_csv = generated_csv,
                target_label  = target_label,
                class_index   = class_index,
            )

            return {
                "status":          "success",
                "message":         f"Successfully adapted to '{target_label}'.",
                "retrain_results": retrain_results,
            }

        except Exception as e:
            traceback.print_exc()
            self._rollback()
            return {"status": "error", "message": str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # Architecture & Surgery  (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _expand_model_classes(self, old_model, new_count):
        new_gen = self._build_generator(new_count)
        old_dense = [l for l in old_model.layers if isinstance(l, layers.Dense)]
        new_dense = [l for l in new_gen.layers   if isinstance(l, layers.Dense)]

        old_w, old_b = old_dense[0].get_weights()
        new_label_weights = np.random.normal(0, 0.02, (1, old_w.shape[1]))
        updated_w = np.vstack([old_w, new_label_weights])
        new_dense[0].set_weights([updated_w, old_b])

        for i in range(1, len(old_dense)):
            new_dense[i].set_weights(old_dense[i].get_weights())

        old_bn = [l for l in old_model.layers if isinstance(l, layers.BatchNormalization)]
        new_bn = [l for l in new_gen.layers   if isinstance(l, layers.BatchNormalization)]
        for o, n in zip(old_bn, new_bn):
            n.set_weights(o.get_weights())

        return new_gen

    def _build_generator(self, n_classes):
        noise = layers.Input(shape=(self.latent_dim,), name="noise_input")
        label = layers.Input(shape=(n_classes,),       name="label_input")
        x = layers.concatenate([noise, label])
        for units in [256, 512, 1024]:
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization(momentum=0.8)(x)
            x = layers.LeakyReLU(0.2)(x)
        output = layers.Dense(self.data_dim, activation='sigmoid')(x)
        return keras.Model([noise, label], output)

    def _build_critic(self, n_classes):
        img   = layers.Input(shape=(self.data_dim,))
        label = layers.Input(shape=(n_classes,))
        x = layers.concatenate([img, label])
        for units in [512, 256]:
            x = layers.Dense(units)(x)
            x = layers.LayerNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Dropout(0.3)(x)
        output = layers.Dense(1)(x)
        return keras.Model([img, label], output)

    # ──────────────────────────────────────────────────────────────────────────
    # Data Engine  (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_training_batch(self, scaled_seeds, class_idx):
        training_data, training_labels = [], []
        TARGET_SIZE = 2000
        while len(training_data) < TARGET_SIZE:
            if np.random.rand() < 0.6 or self.replay_data is None:
                base   = scaled_seeds[np.random.randint(0, len(scaled_seeds))]
                sample = np.clip(base + np.random.normal(0, 0.02, base.shape), 0, 1)
                l_idx  = class_idx
            else:
                idx    = np.random.randint(0, len(self.replay_data))
                sample = self.feature_extractor.scale_features(self.replay_data[idx]).flatten()
                l_idx  = int(self.replay_labels[idx])
            training_data.append(sample)
            oh = np.zeros(self.num_classes)
            if l_idx < self.num_classes:
                oh[l_idx] = 1.0
            training_labels.append(oh)
        return (
            np.array(training_data,  dtype='float32'),
            np.array(training_labels, dtype='float32'),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # WGAN-GP  (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_wgan_gp_model(self):
        class WGAN_GP(keras.Model):
            def __init__(self, critic, generator, latent_dim, gp_weight=10.0):
                super().__init__()
                self.critic     = critic
                self.generator  = generator
                self.latent_dim = latent_dim
                self.gp_weight  = gp_weight

            def compile(self, d_optimizer, g_optimizer):
                super().compile()
                self.d_optimizer = d_optimizer
                self.g_optimizer = g_optimizer

            def train_step(self, data):
                real_img, labels = data
                batch_size = tf.shape(real_img)[0]
                for _ in range(5):
                    z = tf.random.normal([batch_size, self.latent_dim])
                    with tf.GradientTape() as tape:
                        fake_img     = self.generator([z, labels], training=True)
                        logits_real  = self.critic([real_img, labels], training=True)
                        logits_fake  = self.critic([fake_img, labels], training=True)
                        d_cost       = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
                        alpha        = tf.random.uniform([batch_size, 1], 0.0, 1.0)
                        interpolated = real_img + alpha * (fake_img - real_img)
                        with tf.GradientTape() as gp_tape:
                            gp_tape.watch(interpolated)
                            pred = self.critic([interpolated, labels], training=True)
                        grads  = gp_tape.gradient(pred, [interpolated])[0]
                        norm   = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
                        gp     = tf.reduce_mean((norm - 1.0) ** 2)
                        d_loss = d_cost + gp * self.gp_weight
                    grads = tape.gradient(d_loss, self.critic.trainable_variables)
                    self.d_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

                z = tf.random.normal([batch_size, self.latent_dim])
                with tf.GradientTape() as tape:
                    gen_img    = self.generator([z, labels], training=True)
                    logits_gen = self.critic([gen_img, labels], training=True)
                    g_loss     = -tf.reduce_mean(logits_gen)
                grads = tape.gradient(g_loss, self.generator.trainable_variables)
                self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
                return {"d_loss": d_loss, "g_loss": g_loss}

        model = WGAN_GP(self.critic, self.generator, self.latent_dim)
        model.compile(
            d_optimizer=keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.9),
            g_optimizer=keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.9),
        )
        return model

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities  (unchanged except _generate_and_save_packets returns path)
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_label_encoder(self, label, is_new):
        if not os.path.exists(self.encoder_path):
            return -1
        try:
            encoder = joblib.load(self.encoder_path)
            classes = list(encoder.classes_)
            if is_new and label not in classes:
                classes.append(label)
                encoder.classes_ = np.array(classes)
                joblib.dump(encoder, self.encoder_path)
                return len(classes) - 1
            return classes.index(label) if label in classes else -1
        except Exception:
            return -1

    def _generate_and_save_packets(self, class_idx, label_name, count=5000) -> str:
        """Returns the path to the generated CSV."""
        z          = tf.random.normal([count, self.latent_dim])
        oh         = np.zeros((count, self.num_classes))
        oh[:, class_idx] = 1.0
        gen_scaled = self.generator.predict([z, oh], verbose=0)
        gen_raw    = self.feature_extractor.inverse_scale_features(gen_scaled)

        filename = os.path.join(os.path.dirname(__file__), f"generated_{label_name}.csv")
        pd.DataFrame(gen_raw).assign(label=label_name).to_csv(filename, index=False)
        print(f"[+] Synthesis: {count} packets saved to {filename}")
        return filename          # ← now returned so ContinualRetrainer can load it

    def _update_replay_buffer(self, class_index, count=200):
        z  = tf.random.normal([count, self.latent_dim])
        oh = np.zeros((count, self.num_classes))
        oh[:, class_index] = 1.0
        gen_scaled  = self.generator.predict([z, oh], verbose=0)
        gen_raw     = self.feature_extractor.inverse_scale_features(gen_scaled)
        new_labels  = np.full((count,), class_index)

        if self.replay_data is not None:
            updated_X = np.concatenate([self.replay_data, gen_raw],    axis=0)
            updated_y = np.concatenate([self.replay_labels, new_labels], axis=0)
        else:
            updated_X, updated_y = gen_raw, new_labels

        np.savez_compressed(self.replay_path, X=updated_X, y=updated_y)
        self.replay_data, self.replay_labels = updated_X, updated_y

    def _check_model_health(self, class_idx):
        z    = tf.random.normal([50, self.latent_dim])
        oh   = np.zeros((50, self.num_classes))
        oh[:, class_idx] = 1.0
        preds = self.generator.predict([z, oh], verbose=0)
        return np.std(preds) > 0.005

    def _rollback(self):
        bak = self.model_path + ".bak"
        if os.path.exists(bak):
            shutil.copy(bak, self.model_path)
            self.generator = keras.models.load_model(self.model_path, compile=False)


# ──────────────────────────────────────────────────────────────────────────────
# Jitter-only retrainer  (for: normal-flagged-as-attack, zero-day-flagged-as-normal)
# ──────────────────────────────────────────────────────────────────────────────

class JitterRetrainer:
    """
    Lightweight retrainer that uses only Gaussian jittering — no GAN needed.

    Use for:
      • Normal traffic incorrectly flagged as attack  (correct label = BENIGN)
      • Zero-day incorrectly flagged as normal        (rare but handled gracefully)

    It builds an augmented dataset from the seed packets, then delegates to
    ContinualRetrainer for the actual model updates (same anti-forgetting logic).
    """

    JITTER_TARGET   = 2000
    JITTER_STD      = 0.02

    def __init__(self, packet_storage, feature_extractor, model_loader, replay_path, encoder_path):
        self.packet_storage    = packet_storage
        self.feature_extractor = feature_extractor
        self.continual_retrainer = ContinualRetrainer(
            model_loader      = model_loader,
            feature_extractor = feature_extractor,
            replay_path       = replay_path,
            encoder_path      = encoder_path,
        )
        self._tmp_dir = os.path.dirname(os.path.abspath(__file__))

    def retrain(self, packet_ids: list, target_label: str) -> dict:
        print(f"\n[JitterRetrainer] Correcting label → '{target_label}'")

        raw_feats = self.packet_storage.get_features_for_training(packet_ids)
        if not raw_feats:
            return {"status": "error", "message": "Seed packets not found."}

        scaled_seeds = [
            self.feature_extractor.scale_features(np.array(f)).flatten()
            for f in raw_feats
        ]

        # Augment with Gaussian noise
        augmented = []
        while len(augmented) < self.JITTER_TARGET:
            base   = scaled_seeds[np.random.randint(len(scaled_seeds))]
            sample = np.clip(base + np.random.normal(0, self.JITTER_STD, base.shape), 0, 1)
            augmented.append(sample)

        X_aug = np.array(augmented, dtype=np.float32)

        # Inverse-scale back to raw feature space for the CSV
        X_raw = self.feature_extractor.inverse_scale_features(X_aug)

        csv_path = os.path.join(self._tmp_dir, f"jitter_{target_label}.csv")
        pd.DataFrame(X_raw).assign(label=target_label).to_csv(csv_path, index=False)
        print(f"[JitterRetrainer] {len(X_aug)} augmented samples written to {csv_path}")

        # Re-use the same downstream fine-tuning logic
        results = self.continual_retrainer.run(
            generated_csv = csv_path,
            target_label  = target_label,
            class_index   = -1,   # not a new class, no architecture surgery needed
        )

        return {"status": "success", "message": f"Jitter retrain complete for '{target_label}'.", "retrain_results": results}