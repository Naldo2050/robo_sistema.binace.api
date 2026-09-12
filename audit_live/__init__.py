# audit_live — instrumentação forense observacional (NÃO altera lógica de negócio).
"""Pacote de captura forense LIVE.

Ativo somente quando FORENSIC_CAPTURE=1.
Todos os hooks são fire-and-forget, nunca levantam exceção para o caller,
mas NUNCA falham silenciosamente: erros incrementam audit_writer_errors
e são expostos no manifest.json.
"""
