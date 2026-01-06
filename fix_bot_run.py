# fix_bot_run.py - Patch para o método run()
"""
O problema: connection_manager.connect() bloqueia até o WS fechar.
Solução: Executar connect() como uma task paralela e aguardar com timeout.
"""

print("""
CORREÇÃO NECESSÁRIA em market_orchestrator/market_orchestrator.py

No método async def run() (linha ~1869), SUBSTITUIR:

    async def run(self) -> None:
        try:
            self.context_collector.start()
            await self.initialize()
            
            logging.info(...)
            print("═" * 80)
            
            # PROBLEMA: Isto bloqueia indefinidamente
            await self.connection_manager.connect()
            
        except KeyboardInterrupt:
            ...

POR:

    async def run(self) -> None:
        try:
            self.context_collector.start()
            await self.initialize()
            
            logging.info(
                "🎯 Iniciando Enhanced Market Bot v2.3.2 "
                "(modo assíncrono, refatorado em módulos)..."
            )
            print("═" * 80)
            
            # SOLUÇÃO: Executar connect() como task separada
            connect_task = asyncio.create_task(
                self.connection_manager.connect()
            )
            
            # Aguardar até que should_stop seja True
            # ou até connect_task terminar (erro/desconexão)
            while not self.should_stop:
                if connect_task.done():
                    # Conexão encerrada (erro ou desconexão)
                    try:
                        await connect_task  # Re-raise exception se houver
                    except Exception as e:
                        logging.error(f"Connection task falhou: {e}")
                    break
                await asyncio.sleep(0.5)
            
        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        except Exception as e:
            logging.critical(
                f"❌ Erro crítico ao executar o bot: {e}",
                exc_info=True,
            )
        finally:
            # Garante que o gerenciador de conexão pare e feche o WS
            try:
                self.connection_manager.should_stop = True
            except Exception:
                pass

            try:
                await self.connection_manager.disconnect()
            except Exception as e:
                logging.error(
                    f"❌ Erro ao desconectar Connection Manager no run(): {e}",
                    exc_info=True,
                )

            await self.shutdown()
            self._cleanup_handler()
""")