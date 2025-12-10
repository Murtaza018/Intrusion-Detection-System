# gan_retrainer.py
# Fixed: BatchNormalization syntax error (momentum=0.8)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import traceback
import os
import joblib
import pandas as pd

class GanRetrainer:
    def __init__(self, packet_storage, feature_extractor, model_path="Models/cgan_generator.keras"):
        self.packet_storage = packet_storage
        self.feature_extractor = feature_extractor
        self.model_path = model_path
        
        # --- ROBUST PATH SETUP (FIXED) ---
        # Get the directory where THIS script (gan_retrainer.py) is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Now define paths relative to this script, not the terminal command
        self.replay_path = os.path.join(current_dir, "replay_buffer.npz")
        self.encoder_path = os.path.join(current_dir, "label_encoder.pkl")
        
        print(f"[DEBUG] GAN Retrainer looking for files in: {current_dir}")
        
        self.latent_dim = 128
        self.data_dim = 78
        self.num_classes = 15 # Default
        
        # Load Generator
        try:
            self.generator = keras.models.load_model(self.model_path, compile=False)
            print(f"[+] Loaded GAN Generator.")
            # Auto-detect class count from input shape
            self.num_classes = self.generator.input[1].shape[1]
        except:
            print(f"[!] Warning: Could not load GAN. Using default.")
            self.generator = self._build_generator(self.num_classes)

        # Load Replay Buffer
        self.replay_data, self.replay_labels = None, None
        if os.path.exists(self.replay_path):
            try:
                data = np.load(self.replay_path)
                self.replay_data = data['X'] 
                self.replay_labels = data['y']
                print(f"[+] Loaded Replay Buffer: {len(self.replay_data)} samples.")
            except: pass
        else:
            print(f"[!] WARNING: Replay buffer not found at {self.replay_path}")
    def retrain(self, packet_ids, target_label, is_new_label, epochs=50):
        print(f"\n[*] --- STARTING GAN PIPELINE ---")
        print(f"[*] Target Label: '{target_label}' (New: {is_new_label})")
        
        # 1. BACKUP CURRENT MODEL (Safety Net)
        backup_path = self.model_path + ".bak"
        if os.path.exists(self.model_path):
            import shutil
            shutil.copy(self.model_path, backup_path)
            print(f"[*] Backup created at {backup_path}")

        # 2. HANDLE LABELS (Encoder Update)
        class_index = self._handle_label_encoder(target_label, is_new_label)
        if class_index == -1: return {"status": "error", "message": "Label Encoder Error"}

        # 3. MODEL SURGERY (Only if New Label is actually adding a class)
        if is_new_label:
            if class_index >= self.num_classes:
                print(f"[*] Performing Architecture Surgery: {self.num_classes} -> {class_index + 1} classes...")
                self.generator = self._expand_model_classes(self.generator)
                self.num_classes += 1
        
        # Rebuild training parts to match new size
        self.critic = self._build_critic(self.num_classes)
        self.wgan = self._build_wgan_gp()

        # 4. DATA PREP
        raw_feats = self.packet_storage.get_features_for_training(packet_ids)
        if not raw_feats: return {"status": "error", "message": "No features found in DB."}
        
        # Scale inputs to [0,1]
        scaled_feats = [self.feature_extractor.scale_features(np.array(f)).flatten() for f in raw_feats]
        
        training_data = []
        training_labels = []
        TARGET_TOTAL = 1000
        
        print(f"[*] Augmenting {len(scaled_feats)} packets to {TARGET_TOTAL} synthetic samples...")
        
        while len(training_data) < TARGET_TOTAL:
            # 60% Bias towards the input packets
            if np.random.rand() < 0.6 or self.replay_data is None:
                base = scaled_feats[np.random.randint(0, len(scaled_feats))]
                label_idx = class_index
                noise = np.random.normal(0, 0.02, base.shape)
                sample = np.clip(base + noise, 0.0, 1.0)
            else:
                # 40% Replay
                idx = np.random.randint(0, len(self.replay_data))
                raw_sample = self.replay_data[idx]
                sample = self.feature_extractor.scale_features(raw_sample).flatten()
                label_idx = int(self.replay_labels[idx])
            
            training_data.append(sample)
            
            # One-hot encoding
            one_hot = np.zeros(self.num_classes)
            if label_idx < self.num_classes:
                one_hot[label_idx] = 1.0
            training_labels.append(one_hot)

        X_train = np.array(training_data, dtype='float32')
        y_train = np.array(training_labels, dtype='float32')

        # 5. TRAIN
        print(f"[*] Fine-tuning GAN on {len(X_train)} samples...")
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(64)
        
        try:
            self.wgan.fit(dataset, epochs=epochs, verbose=0)
            
            # --- POST-TRAINING SAFETY CHECK ---
            print("[*] Verifying model stability before saving...")
            if self._check_model_health(class_index):
                # SUCCESS
                self.generator.save(self.model_path)
                
                # 6. GENERATE & SAVE
                gen_count = 5000
                print(f"[*] Generating {gen_count} fresh packets for class '{target_label}'...")
                self._generate_and_save_packets(class_index, target_label, count=gen_count)
                
                # 7. UPDATE BUFFER
                self._update_replay_buffer(class_index)
                
                return {"status": "success", "message": f"Trained on '{target_label}' & Verified."}
            else:
                # FAILURE
                print("[!] CRITICAL: Mode Collapse detected. Reverting to backup.")
                if os.path.exists(backup_path):
                    import shutil
                    shutil.copy(backup_path, self.model_path)
                    self.generator = keras.models.load_model(self.model_path, compile=False)
                
                return {"status": "error", "message": "Training failed (Mode Collapse). Reverted to previous model."}

        except Exception as e:
            traceback.print_exc()
            if os.path.exists(backup_path): 
                import shutil
                shutil.copy(backup_path, self.model_path)
            return {"status": "error", "message": str(e)}

    # --- HELPERS ---
    def _handle_label_encoder(self, label, is_new):
        if not os.path.exists(self.encoder_path):
            print("[!] Encoder not found!"); return -1
        try:
            encoder = joblib.load(self.encoder_path)
            current_classes = list(encoder.classes_)
            if is_new:
                if label in current_classes:
                    print(f"[*] Label '{label}' already exists. Switching to Update Mode.")
                    return list(encoder.transform([label]))[0]
                print(f"[*] Registering NEW label: '{label}'")
                new_classes = current_classes + [label]
                encoder.classes_ = np.array(new_classes)
                joblib.dump(encoder, self.encoder_path)
                return len(new_classes) - 1
            else:
                if label not in current_classes:
                    print(f"[!] Error: Label '{label}' not found in encoder.")
                    return -1
                return list(encoder.transform([label]))[0]
        except Exception as e:
            print(f"[!] Encoder Error: {e}"); return -1

    def _generate_and_save_packets(self, class_index, label_name, count=5000):
        noise = tf.random.normal(shape=(count, self.latent_dim))
        labels = np.zeros((count, self.num_classes))
        labels[:, class_index] = 1.0
        gen_scaled = self.generator.predict([noise, labels], verbose=0)
        gen_raw = self.feature_extractor.inverse_scale_features(gen_scaled)
        output_file = f"generated_{label_name}.csv"
        df = pd.DataFrame(gen_raw)
        df['label'] = label_name 
        df.to_csv(output_file, index=False)
        print(f"[+] Saved generated packets to {output_file}")

    def _update_replay_buffer(self, class_index, count=200):
        print(f"[*] Updating Replay Buffer with Class {class_index}...")
        noise = tf.random.normal(shape=(count, self.latent_dim))
        labels = np.zeros((count, self.num_classes))
        labels[:, class_index] = 1.0
        gen_scaled = self.generator.predict([noise, labels], verbose=0)
        gen_raw = self.feature_extractor.inverse_scale_features(gen_scaled)
        new_labels = np.full((count,), class_index)
        
        if self.replay_data is not None:
            updated_X = np.concatenate([self.replay_data, gen_raw], axis=0)
            updated_y = np.concatenate([self.replay_labels, new_labels], axis=0)
        else:
            updated_X, updated_y = gen_raw, new_labels
            
        try:
            np.savez_compressed(self.replay_path, X=updated_X, y=updated_y)
            self.replay_data, self.replay_labels = updated_X, updated_y
            print(f"[+] Buffer Saved. New size: {len(updated_X)}")
        except Exception as e: print(f"[!] Buffer Save Error: {e}")

    def _check_model_health(self, class_idx):
        try:
            noise = tf.random.normal(shape=(50, self.latent_dim))
            labels = np.zeros((50, self.num_classes))
            labels[:, class_idx] = 1.0
            preds = self.generator.predict([noise, labels], verbose=0)
            diversity = np.std(preds)
            print(f"    > Diversity Score: {diversity:.5f}")
            if diversity < 0.01:
                print("    [!] FAILED: Low diversity.")
                return False
            if np.min(preds) < 0.0 or np.max(preds) > 1.0:
                print("    [!] FAILED: Output out of bounds.")
                return False
            print("    [+] Model health looks good.")
            return True
        except Exception as e:
            print(f"    [!] Health check crashed: {e}")
            return False

    # --- ARCHITECTURE (FIXED BATCHNORM) ---
    def _expand_model_classes(self, old_model):
        old_input_label = old_model.input[1]
        new_num_classes = old_input_label.shape[1] + 1
        new_input_label = layers.Input(shape=(new_num_classes,), name="new_label_input")
        input_noise = layers.Input(shape=(self.latent_dim,), name="noise_input")
        x = layers.concatenate([input_noise, new_input_label])
        
        # FIX: Explicitly name momentum argument
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization(momentum=0.8)(x) # <--- FIXED
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization(momentum=0.8)(x) # <--- FIXED
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization(momentum=0.8)(x) # <--- FIXED
        x = layers.LeakyReLU(0.2)(x)
        
        output = layers.Dense(self.data_dim, activation='sigmoid')(x)
        new_model = keras.Model([input_noise, new_input_label], output, name="generator_expanded")
        
        # Weight Copy Logic
        old_dense_1 = [l for l in old_model.layers if isinstance(l, layers.Dense)][0]
        new_dense_1 = [l for l in new_model.layers if isinstance(l, layers.Dense)][0]
        old_w, old_b = old_dense_1.get_weights()
        padding = np.random.normal(0, 0.02, (1, 256)) 
        new_w = np.vstack([old_w, padding])
        new_dense_1.set_weights([new_w, old_b])
        
        old_layers = [l for l in old_model.layers if len(l.weights) > 0 and l != old_dense_1]
        new_layers = [l for l in new_model.layers if len(l.weights) > 0 and l != new_dense_1]
        for o_l, n_l in zip(old_layers, new_layers): n_l.set_weights(o_l.get_weights())
        return new_model

    def _build_critic(self, n_classes):
        img = layers.Input(shape=(self.data_dim,))
        label = layers.Input(shape=(n_classes,))
        x = layers.concatenate([img, label])
        x = layers.Dense(512)(x); x = layers.LeakyReLU(0.2)(x); x = layers.Dropout(0.3)(x)
        x = layers.Dense(256)(x); x = layers.LeakyReLU(0.2)(x); x = layers.Dropout(0.3)(x)
        x = layers.Dense(128)(x); x = layers.LeakyReLU(0.2)(x)
        output = layers.Dense(1)(x)
        return keras.Model([img, label], output, name="critic")

    def _build_generator(self, n_classes):
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(n_classes,))
        x = layers.concatenate([noise, label])
        
        # FIX: Explicitly name momentum argument
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization(momentum=0.8)(x) # <--- FIXED
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization(momentum=0.8)(x) # <--- FIXED
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization(momentum=0.8)(x) # <--- FIXED
        x = layers.LeakyReLU(0.2)(x)
        
        output = layers.Dense(self.data_dim, activation='sigmoid')(x)
        return keras.Model([noise, label], output, name="generator")

    def _build_wgan_gp(self):
        class WGAN_GP_Model(keras.Model):
            def __init__(self, critic, generator, latent_dim, d_steps=5, gp_weight=10.0):
                super(WGAN_GP_Model, self).__init__()
                self.critic = critic
                self.generator = generator
                self.latent_dim = latent_dim
                self.d_steps = d_steps
                self.gp_weight = gp_weight
            def gradient_penalty(self, batch_size, real, fake, labels):
                alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
                diff = fake - real
                interpolated = real + alpha * diff
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    pred = self.critic([interpolated, labels], training=True)
                grads = gp_tape.gradient(pred, [interpolated])[0]
                norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
                return tf.reduce_mean((norm - 1.0) ** 2)
            def train_step(self, data):
                real_images, labels = data
                batch_size = tf.shape(real_images)[0]
                for i in range(self.d_steps):
                    noise = tf.random.normal([batch_size, self.latent_dim])
                    with tf.GradientTape() as tape:
                        fake_images = self.generator([noise, labels], training=True)
                        fake_logits = self.critic([fake_images, labels], training=True)
                        real_logits = self.critic([real_images, labels], training=True)
                        d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                        gp = self.gradient_penalty(batch_size, real_images, fake_images, labels)
                        d_loss = d_cost + gp * self.gp_weight
                    d_grad = tape.gradient(d_loss, self.critic.trainable_variables)
                    self.d_optimizer.apply_gradients(zip(d_grad, self.critic.trainable_variables))
                noise = tf.random.normal([batch_size, self.latent_dim])
                with tf.GradientTape() as tape:
                    fake_images = self.generator([noise, labels], training=True)
                    gen_logits = self.critic([fake_images, labels], training=True)
                    g_loss = -tf.reduce_mean(gen_logits)
                g_grad = tape.gradient(g_loss, self.generator.trainable_variables)
                self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))
                return {"d_loss": d_loss, "g_loss": g_loss}
        model = WGAN_GP_Model(self.critic, self.generator, self.latent_dim)
        model.compile(d_optimizer=keras.optimizers.Adam(0.0002, 0.0, 0.9), g_optimizer=keras.optimizers.Adam(0.0002, 0.0, 0.9))
        return model