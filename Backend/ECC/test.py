from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# Load your private key
with open("key.pem", "rb") as f:
    private_key = load_pem_private_key(f.read(), password=None)

# Export public key coordinates
pub_numbers = private_key.public_key().public_numbers()

x_hex = pub_numbers.x.to_bytes(32, 'big').hex()
y_hex = pub_numbers.y.to_bytes(32, 'big').hex()

print("X:", x_hex)
print("Y:", y_hex)
print("X length:", len(x_hex), "chars")  # Must be 64
print("Y length:", len(y_hex), "chars")  # Must be 64