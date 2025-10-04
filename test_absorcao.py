import math

# importe do seu módulo
from flow_analyzer import FlowAnalyzer

def test_absorcao_por_sinal_delta():
    fa = FlowAnalyzer()
    eps = getattr(fa, "absorcao_eps", 1.0)

    casos = [
        (-35.57, "Absorção de Venda"),
        (+7.53,  "Absorção de Compra"),
        (+1.50,  "Absorção de Compra"),
        (0.0,    "Neutra"),
        (eps/2,  "Neutra"),
        (-(eps/2),"Neutra"),
        (eps*1.1,"Absorção de Compra"),
        (-eps*1.1,"Absorção de Venda"),
    ]
    for delta, esperado in casos:
        rotulo = fa.classificar_absorcao_por_delta(delta, eps=eps)
        assert rotulo == esperado, f"delta={delta} → {rotulo}, esperado={esperado}"
