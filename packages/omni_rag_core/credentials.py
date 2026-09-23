from cryptography.fernet import Fernet, InvalidToken


class CredentialEncryptionError(ValueError):
    pass


class CredentialCipher:
    def __init__(self, key: str | None):
        self._fernet: Fernet | None = None
        if key:
            try:
                self._fernet = Fernet(key.encode("utf-8"))
            except (TypeError, ValueError) as exc:
                raise CredentialEncryptionError(
                    "LLM_CREDENTIALS_KEY must be a valid Fernet key"
                ) from exc

    def encrypt(self, value: str) -> str:
        if not self._fernet:
            raise CredentialEncryptionError(
                "LLM_CREDENTIALS_KEY is required before saving provider API keys"
            )
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str) -> str:
        if not self._fernet:
            raise CredentialEncryptionError(
                "LLM_CREDENTIALS_KEY is required before using provider API keys"
            )
        try:
            return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeError, ValueError) as exc:
            raise CredentialEncryptionError(
                "Stored provider API key could not be decrypted"
            ) from exc
