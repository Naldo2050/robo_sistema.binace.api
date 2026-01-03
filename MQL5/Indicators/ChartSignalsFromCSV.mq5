//+------------------------------------------------------------------+
//|           ChartSignalsFromCSV.mq5                                |
//|   Lê mt5_signals\signals.csv e desenha setas no gráfico          |
//+------------------------------------------------------------------+
#property copyright "2026, MetaQuotes Software"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

// Parâmetro de entrada: nome do arquivo (relativo à pasta MQL5\Files)
input string InpFileName = "mt5_signals\\signals.csv";

// Variável global para controlar o último sinal processado
datetime last_signal_time = 0;

//+------------------------------------------------------------------+
//| Inicialização                                                     |
//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(1); // chama OnTimer a cada 1 segundo
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Finalização                                                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

//+------------------------------------------------------------------+
//| Timer                                                             |
//+------------------------------------------------------------------+
void OnTimer()
  {
   ProcessSignalsFromCsv();
  }

//+------------------------------------------------------------------+
//| Leitura e processamento do CSV                                   |
//+------------------------------------------------------------------+
void ProcessSignalsFromCsv()
  {
   int handle = FileOpen(InpFileName,
                         FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;

   // Ler e descartar o cabeçalho (14 colunas)
   string dummy;
   for(int i = 0; i < 14 && !FileIsEnding(handle); i++)
      dummy = FileReadString(handle);

   // Ler cada linha
   while(!FileIsEnding(handle))
     {
      string ts_utc     = FileReadString(handle); if(FileIsEnding(handle)) break;
      string symbol_csv = FileReadString(handle); if(FileIsEnding(handle)) break;
      string exchange   = FileReadString(handle); if(FileIsEnding(handle)) break;
      string event_type = FileReadString(handle); if(FileIsEnding(handle)) break;
      string side       = FileReadString(handle); if(FileIsEnding(handle)) break;
      string s_price    = FileReadString(handle); if(FileIsEnding(handle)) break;
      string s_delta    = FileReadString(handle); if(FileIsEnding(handle)) break;
      string s_volume   = FileReadString(handle); if(FileIsEnding(handle)) break;
      string s_poc      = FileReadString(handle); if(FileIsEnding(handle)) break;
      string s_val      = FileReadString(handle); if(FileIsEnding(handle)) break;
      string s_vah      = FileReadString(handle); if(FileIsEnding(handle)) break;
      string regime     = FileReadString(handle); if(FileIsEnding(handle)) break;
      string strength   = FileReadString(handle); if(FileIsEnding(handle)) break;
      string context    = FileReadString(handle); // última coluna

      if(ts_utc == "")
         continue;

      // Converte timestamp ("YYYY.MM.DD HH:MM:SS") para datetime
      datetime t = StringToTime(ts_utc);
      if(t <= 0)
         continue;

      // Ignora sinais antigos já processados
      if(last_signal_time != 0 && t <= last_signal_time)
         continue;

      // Garante que o símbolo do CSV seja o mesmo do gráfico
      if(symbol_csv != _Symbol)
         continue;

      double price = StringToDouble(s_price);

      // Encontra a barra mais próxima ao timestamp
      int shift = iBarShift(_Symbol, PERIOD_CURRENT, t, true);
      if(shift < 0)
         continue;

      datetime bar_time = iTime(_Symbol, PERIOD_CURRENT, shift);

      // Define cor e tipo de seta conforme o side
      color clr = clrYellow;
      int arrowCode = 159; // seta genérica
      if(side == "buy")
        {
         clr       = clrLime;
         arrowCode = 241; // seta para cima
        }
      else if(side == "sell")
        {
         clr       = clrRed;
         arrowCode = 242; // seta para baixo
        }

      // Nome simples baseado no timestamp em segundos
      string name = "sig_" + IntegerToString((long)t);

      if(ObjectFind(0, name) == -1)
        {
         if(ObjectCreate(0, name, OBJ_ARROW, 0, bar_time, price))
           {
            ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
            ObjectSetInteger(0, name, OBJPROP_ARROWCODE, arrowCode);
            ObjectSetString(0, name, OBJPROP_TEXT, event_type + " | " + strength);
           }
        }

      if(t > last_signal_time)
         last_signal_time = t;
     }

   FileClose(handle);
  }

//+------------------------------------------------------------------+
//| Cálculo do indicador (não usamos buffers)                        |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
  {
   return(rates_total);
  }
//+------------------------------------------------------------------+