import os
import base64

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as sym_padding

def _load_private_key():
    pem = os.environ["RSA_PRIVATE_KEY"].encode("utf-8")
    return serialization.load_pem_private_key(pem, password=None)

def decrypt_session_key(encrypted_session_key_b64: str) -> bytearray:
    raw = base64.b64decode(encrypted_session_key_b64)
    key = _load_private_key()
    aes_key = key.decrypt(
        raw,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return bytearray(aes_key)

def decrypt_video(aes_key: bytearray, encrypted_blob: bytes) -> bytearray:
    iv = encrypted_blob[:16]
    ciphertext = encrypted_blob[16:]

    decryptor = Cipher(algorithms.AES(bytes(aes_key)), modes.CBC(iv)).decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = sym_padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return bytearray(plaintext)

def shred(buf:bytearray):
    buf[:] = b'\x00' * len(buf)
