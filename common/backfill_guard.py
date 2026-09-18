# common/backfill_guard.py
# -*- coding: utf-8 -*-
"""
Guarda anti-contaminação backfill/replay x produção ao vivo (AÇÃO 2, follow-up
da observação go-live 18/09/2026; contexto: INV-A em docs/audit/TEST_DEBT.md).

Achado INV-A: 3.271 sinais `Absorção` slim (`data_context=historical`) no
`dados/trading_bot.db` vieram de replays/backfills gravados no MESMO banco do
bot ao vivo — nenhum emissor vivo os produz. Para que isso não se repita,
qualquer script que escreva dados históricos no DB de produção deve chamar
`enforce_no_live_bot()` no início e abortar se `main.py` estiver ativo.

Alternativa estrutural preferível (quando aplicável): apontar o backfill para
`dados/trading_bot_historical.db` em vez do banco de produção.
"""
from __future__ import annotations

from typing import Iterable, Optional

LIVE_BOT_MARKER = "main.py"
PROD_DB_PATH = "dados/trading_bot.db"
HISTORICAL_DB_PATH = "dados/trading_bot_historical.db"


def _iter_cmdlines() -> Iterable[str]:
    """Lê as linhas de comando dos processos ativos (lazy import de psutil).

    Mantido para compatibilidade; a varredura real usa tokens estruturados.
    Exclui o próprio processo.
    """
    import os

    import psutil

    me = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info.get("pid") == me:
                continue
            cmdline = proc.info.get("cmdline") or []
        except Exception:
            continue
        yield " ".join(cmdline)


def _tokens_match(tokens) -> bool:
    """True se os tokens indicam `python ... main.py` como script executado.

    Regra: executável python* + primeiro argumento posicional (após flags como
    `-u`; pulando o código de `-c` e abortando em `-m`) com basename main.py.
    Só o ARGV[script] conta — menções em outros tokens (shell que referencia
    um caminho main.py, editores, `python -c "..."`) não disparam.
    """
    import os

    toks = [t.strip("\"'") for t in tokens]
    if not toks:
        return False
    exe = os.path.basename(toks[0]).lower()
    if not (exe.startswith("python") or exe == "py"):
        return False
    i = 1
    while i < len(toks):
        t = toks[i]
        if t == "-c":
            return False  # código inline, nunca main.py
        if t == "-m":
            return False  # módulo, nunca o script main.py
        if t.startswith("-"):
            i += 1
            continue
        return os.path.basename(t).lower() == LIVE_BOT_MARKER
    return False


def is_live_bot_running(cmdlines: Optional[Iterable[str]] = None) -> bool:
    """True se algum processo ativo executa `main.py` como script."""
    if cmdlines is not None:
        return any(_tokens_match(cmd.split()) for cmd in cmdlines)
    import os

    import psutil

    me = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info.get("pid") == me:
                continue
            if _tokens_match(proc.info.get("cmdline") or []):
                return True
        except Exception:
            continue
    return False


def enforce_no_live_bot(
    db_path: str = PROD_DB_PATH,
    cmdlines: Optional[Iterable[str]] = None,
) -> None:
    """Aborta com RuntimeError se o bot ao vivo estiver ativo.

    `cmdlines` existe para testabilidade (injeção de lista fake); em produção
    a varredura real via psutil é usada.
    """
    if is_live_bot_running(cmdlines):
        raise RuntimeError(
            "Backfill/replay abortado: main.py está ativo. "
            "Use um banco separado ou aguarde o encerramento da sessão ao vivo."
        )
