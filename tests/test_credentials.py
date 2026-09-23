import pytest
from cryptography.fernet import Fernet

from packages.omni_rag_core.credentials import CredentialCipher, CredentialEncryptionError


def test_credential_cipher_encrypts_and_decrypts_without_exposing_plaintext():
    cipher = CredentialCipher(Fernet.generate_key().decode())

    encrypted = cipher.encrypt("company-secret")

    assert encrypted != "company-secret"
    assert cipher.decrypt(encrypted) == "company-secret"


def test_credential_cipher_requires_valid_key_and_rejects_invalid_ciphertext():
    with pytest.raises(CredentialEncryptionError, match="valid Fernet key"):
        CredentialCipher("not-a-fernet-key")

    unavailable = CredentialCipher(None)
    with pytest.raises(CredentialEncryptionError, match="required before saving"):
        unavailable.encrypt("secret")
    with pytest.raises(CredentialEncryptionError, match="required before using"):
        unavailable.decrypt("ciphertext")

    cipher = CredentialCipher(Fernet.generate_key().decode())
    with pytest.raises(CredentialEncryptionError, match="could not be decrypted"):
        cipher.decrypt("not-ciphertext")
