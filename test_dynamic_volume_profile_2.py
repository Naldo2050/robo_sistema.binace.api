import pandas as pd
import numpy as np
from dynamic_volume_profile import DynamicVolumeProfile

def test_poc_calculation():
    print("Testando cálculo de POC...")
    
    np.random.seed(42)
    n_trades = 1000
    prices = np.random.normal(100, 1, n_trades)
    quantities = np.random.uniform(1, 10, n_trades)
    sides = np.random.choice([1, -1], n_trades)
    
    df = pd.DataFrame({
        'p': prices,
        'q': quantities,
        'm': sides
    })
    
    vp = DynamicVolumeProfile('BTC/USDT', base_bins=20)
    result = vp.calculate(df, atr=0.5, whale_buy_volume=1000, whale_sell_volume=800, cvd=500)
    
    print(f"POC: {result['poc_price']:.2f}")
    assert result['status'] == 'success'
    assert result['poc_price'] > 0
    print("✅ Teste de POC passou!")

def test_volume_profile_creation():
    print("\nTestando criação de Volume Profile...")
    
    np.random.seed(0)
    n_trades = 500
    prices = np.random.normal(50, 0.5, n_trades)
    quantities = np.random.uniform(1, 5, n_trades)
    sides = np.random.choice([1, -1], n_trades)
    
    df = pd.DataFrame({
        'p': prices,
        'q': quantities,
        'm': sides
    })
    
    vp = DynamicVolumeProfile('ETH/USDT', base_bins=15)
    result = vp.calculate(df, atr=0.2, whale_buy_volume=500, whale_sell_volume=400, cvd=200)
    
    assert result['status'] == 'success'
    assert len(result['hvns']) > 0 or len(result['lvns']) > 0
    assert result['vah'] > result['val']
    print("✅ Teste de Volume Profile passou!")

def test_empty_dataframe():
    print("\nTestando DataFrame vazio...")
    
    df = pd.DataFrame(columns=['p', 'q', 'm'])
    
    vp = DynamicVolumeProfile('BTC/USDT', base_bins=20)
    result = vp.calculate(df)
    
    assert result['status'] == 'error'
    print("✅ Teste de DataFrame vazio passou!")

if __name__ == "__main__":
    try:
        test_poc_calculation()
        test_volume_profile_creation()
        test_empty_dataframe()
        print("\n✅ Todos os testes passaram com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        print(traceback.format_exc())
