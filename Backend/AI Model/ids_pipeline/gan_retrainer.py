# gan_retrainer.py
# Advanced GAN Fine-Tuning with Continual Learning (Buffer Updates)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import traceback
import os

class GanRetrainer:
    def __init__(self, packet_storage, feature_extractor, model_path="Models/cgan_generator.keras"):
        self.packet_storage = packet_storage
        self.feature_extractor = feature_extractor
        self.model_path = model_path
        self.replay_path = "ids_pipeline/replay_buffer.npz"
        self.latent_dim = 128
        self.data_dim = 78
        self.num_classes = 15 
        
        try:
            self.generator = keras.models.load_model(self.model_path, compile=False)
            print(f"[+] Loaded GAN Generator from {self.model_path}")
            self.num_classes = self.generator.input[1].shape[1]
            print(f"[*] Detected {self.num_classes} existing classes.")
        except:
            print(f"[!] Warning: Could not load GAN. Starting fresh.")
            self.generator = self._build_generator(self.num_classes)

        # Load Replay Buffer
        self.replay_data = None
        self.replay_labels = None
        if os.path.exists(self.replay_path):
            try:
                data = np.load(self.replay_path)
                self.replay_data = data['X']
                self.replay_labels = data['y']
            except Exception as e:
                print(f"[!] Error loading replay buffer: {e}")

    def retrain(self, zero_day_ids, epochs=50):
        print(f"[*] Starting GAN Retraining Process...")
        
        # 1. MODEL SURGERY
        new_class_index = self.num_classes
        print(f"[*] Expanding Architecture: {self.num_classes} -> {new_class_index + 1} classes...")
        self.generator = self._expand_model_classes(self.generator)
        self.num_classes += 1
        
        self.critic = self._build_critic(self.num_classes)
        self.wgan = self._build_wgan_gp()

        # 2. DATA PREP
        # (Fetch new zero-day features)
        raw_zd = self.packet_storage.get_features_for_training(zero_day_ids)
        if not raw_zd: return {"status": "error", "message": "No valid features found."}
        scaled_zd = [self.feature_extractor.scale_features(np.array(f)).flatten() for f in raw_zd]
        
        # (Fetch Replay Data from file)
        replay_feats = []
        if self.replay_data is not None:
            # We assume replay buffer contains SCALED or UNSCALED data?
            # Ideally, store UNSCALED in buffer to be safe, so we scale here.
            # If your generate_replay_buffer.py saved raw X_full, we must scale.
            indices = np.random.choice(len(self.replay_data), min(2000, len(self.replay_data)), replace=False)
            for idx in indices:
                raw = self.replay_data[idx]
                label = self.replay_labels[idx]
                scaled = self.feature_extractor.scale_features(raw).flatten()
                replay_feats.append((scaled, int(label)))

        # 3. AUGMENTATION
        training_data = []
        training_labels = []
        TARGET_TOTAL = 1000
        
        while len(training_data) < TARGET_TOTAL:
            if np.random.rand() < 0.5 or not replay_feats:
                # Train on NEW Zero-Day
                base = scaled_zd[np.random.randint(0, len(scaled_zd))]
                label_idx = new_class_index
                noise = np.random.normal(0, 0.02, base.shape)
                sample = np.clip(base + noise, 0.0, 1.0)
            else:
                # Train on OLD Replay
                base, label_idx = replay_feats[np.random.randint(0, len(replay_feats))]
                sample = base # Already scaled
            
            training_data.append(sample)
            one_hot = np.zeros(self.num_classes)
            one_hot[label_idx] = 1.0
            training_labels.append(one_hot)

        X_train = np.array(training_data, dtype='float32')
        y_train = np.array(training_labels, dtype='float32')

        # 4. TRAIN
        print(f"[*] Training on {len(X_train)} samples...")
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(64)
        
        try:
            self.wgan.fit(dataset, epochs=epochs, verbose=0)
            self.generator.save(self.model_path)
            
            # --- CRITICAL STEP: Update Replay Buffer ---
            self._update_replay_buffer(new_class_index)
            # -------------------------------------------
            
            return {"status": "success", "message": f"Retrained on Class {new_class_index}. Buffer updated."}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _update_replay_buffer(self, class_index, count=200):
        """
        Generates synthetic samples of the NEW class and saves them to the replay buffer.
        This prevents 'Catastrophic Forgetting' in future cycles.
        """
        print(f"[*] Generating {count} replay samples for new Class {class_index}...")
        
        # 1. Generate Synthetic Data
        noise = tf.random.normal(shape=(count, self.latent_dim))
        labels = np.zeros((count, self.num_classes))
        labels[:, class_index] = 1.0
        
        generated_scaled = self.generator.predict([noise, labels], verbose=0)
        
        # 2. Inverse Scale (Convert 0.0-1.0 back to Raw Values)
        # We store RAW values in the buffer so future scalers work correctly
        generated_raw = self.feature_extractor.inverse_scale_features(generated_scaled)
        
        # 3. Create Label Array
        new_labels = np.full((count,), class_index)
        
        # 4. Append to existing buffer
        if self.replay_data is not None:
            updated_X = np.concatenate([self.replay_data, generated_raw], axis=0)
            updated_y = np.concatenate([self.replay_labels, new_labels], axis=0)
        else:
            updated_X = generated_raw
            updated_y = new_labels
            
        # 5. Save back to disk
        try:
            np.savez_compressed(self.replay_path, X=updated_X, y=updated_y)
            print(f"[+] Replay Buffer Updated! New Size: {len(updated_X)}")
            
            # Reload into memory
            self.replay_data = updated_X
            self.replay_labels = updated_y
        except Exception as e:
            print(f"[!] Failed to update buffer: {e}")

    # --- (Include _expand_model_classes, _build_critic, _build_generator, _build_wgan_gp here) ---
    # (Copy these from the previous message, they haven't changed)
    def _expand_model_classes(self, old_model):
        old_input_label = old_model.input[1]
        new_num_classes = old_input_label.shape[1] + 1
        new_input_label = layers.Input(shape=(new_num_classes,), name="new_label_input")
        input_noise = layers.Input(shape=(self.latent_dim,), name="noise_input")
        x = layers.concatenate([input_noise, new_input_label])
        x = layers.Dense(256)(x); x = layers.BatchNormalization(0.8)(x); x = layers.LeakyReLU(0.2)(x)
        x = layers.Dense(512)(x); x = layers.BatchNormalization(0.8)(x); x = layers.LeakyReLU(0.2)(x)
        x = layers.Dense(1024)(x); x = layers.BatchNormalization(0.8)(x); x = layers.LeakyReLU(0.2)(x)
        output = layers.Dense(self.data_dim, activation='sigmoid')(x)
        new_model = keras.Model([input_noise, new_input_label], output, name="generator_expanded")
        
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
        x = layers.Dense(256)(x); x = layers.BatchNormalization(0.8)(x); x = layers.LeakyReLU(0.2)(x)
        x = layers.Dense(512)(x); x = layers.BatchNormalization(0.8)(x); x = layers.LeakyReLU(0.2)(x)
        x = layers.Dense(1024)(x); x = layers.BatchNormalization(0.8)(x); x = layers.LeakyReLU(0.2)(x)
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