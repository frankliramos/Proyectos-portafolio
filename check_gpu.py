import torch
import xgboost as xgb
import sys
import platform

def print_tech_stack():
    print("🚀 --- REPORTE TÉCNICO DE ENTORNO --- 🚀")
    print(f"Sistema Operativo: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]} en {sys.executable}")
    print("-" * 40)
    
    # Verificación PyTorch + CUDA
    cuda_available = torch.cuda.is_available()
    print(f"¿PyTorch detecta CUDA?: {'✅ SÍ' if cuda_available else '❌ NO'}")
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {device_name}")
        print(f"Arquitectura CUDA (Capability): {capability[0]}.{capability[1]}")
        print(f"Versión CUDA en PyTorch: {torch.version.cuda}")
        
        # Prueba de tensor en GPU
        x = torch.randn(1).to("cuda")
        print("Prueba de Tensor en GPU: ✅ Exitosa")
    
    print("-" * 40)
    
    # Verificación XGBoost
    print(f"XGBoost Versión: {xgb.__version__}")
    try:
        # Intento de crear un DMatrix en GPU para validar soporte
        import numpy as np
        data = np.random.rand(5, 5)
        label = np.random.randint(2, size=5)
        dtrain = xgb.DMatrix(data, label=label)
        params = {'tree_method': 'hist', 'device': 'cuda'}
        xgb.train(params, dtrain, num_boost_round=1)
        print("XGBoost con soporte GPU: ✅ Activo")
    except Exception as e:
        print(f"XGBoost con soporte GPU: ⚠️ No disponible ({e})")

if __name__ == "__main__":
    print_tech_stack()