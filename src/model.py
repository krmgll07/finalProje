"""
Personel Normu Tahmini için Derin Öğrenme Modeli
==================================================
Bu modül, Türk tarım bölgelerinde optimal personel tahsisini tahmin etmek için sinir ağı mimarilerini tanımlar.

Modeller:

- PersonnelMLP: Regresyon için Çok Katmanlı Perceptron
- PersonnelResNet: Atlamalı bağlantılara sahip Artık Ağ
- PersonnelAttentionNet: Dikkat mekanizmalı ağ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#Girdi özelliklerini ardışık doğrusal katmanlar, 
# BatchNorm, ReLU ve Dropout kullanarak işler ve sonunda tek bir regresyon çıktısı üretir. 
# Amaç, personel normunu tahmin etmek için klasik bir çok katmanlı sinir ağı (MLP) kurmaktır.
class PersonnelMLP(nn.Module):
    """
    Personel normu tahmini için Çok Katmanlı Perceptron.

    Yapılandırılabilir gizli katmanlara sahip ileri beslemeli sinir ağı,

    düzenleme için dropout ve toplu normalizasyon.

    Mimari:

    Giriş -> [Doğrusal -> Toplu Normalizasyon -> ReLU -> Dropout] x N -> Çıkış

    """
#Modelin katmanlarını hidden_dims listesine göre dinamik şekilde oluşturur ve son katmanda tek değer döndüren bir çıkış katmanı ekler.
# Böylece ağın derinliği ve katman boyutları kolayca ayarlanabilir.
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.3):
        """
        Initialize the MLP model.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
        """
        super(PersonnelMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
#modele verdiğin giriş verisini (x) alır ve katmanların içinden geçirip tahmini üretir.
# En sonda squeeze(-1) ile çıktının şekli (batch_size, 1) yerine (batch_size) olur, yani daha “düz” bir çıktı verir.
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.network(x).squeeze(-1)

#ResNet mantığıyla çalışan bir artık blok tanımlar ve girişe skip connection ekleyerek daha stabil eğitim sağlar.
#  Derin ağlarda kaybolan gradyan problemini azaltır.
class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Architecture:
        x -> Linear -> BN -> ReLU -> Linear -> BN -> (+x) -> ReLU
    """
#Aynı boyutta iki doğrusal katman (Linear + BN) oluşturur ve araya aktivasyon ile dropout ekler. 
# Bu yapı, giriş boyutunu değiştirmeden “blok” gibi tekrar kullanılabilir hale getirir.
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        """
        Initialize residual block.

        Args:
            dim: Dimension of the block (input and output)
            dropout_rate: Dropout probability
        """
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
#Girişi bloktan geçirip sonucu girişle toplar (skip connection) ve ardından ReLU uygular. 
# Böylece ağ, hem öğrenilen dönüşümü hem de orijinal sinyali birlikte kullanır.
    def forward(self, x):
        """Forward pass with skip connection."""
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out

#Girdiyi önce hidden_dim boyutuna projekte eder, ardından birkaç ResidualBlock ile işler ve regresyon çıktısı üretir.
# Bu yapı MLP’ye göre daha derin eğitime uygundur ve daha iyi genelleme sağlayabilir.
class PersonnelResNet(nn.Module):
    """
    Residual Network for personnel norm prediction.

    Uses skip connections to enable training of deeper networks
    and prevent vanishing gradients.

    Architecture:
        Input -> Linear -> [ResidualBlock] x N -> Linear -> Output
    """
#İlk katmanda giriş özelliklerini hidden_dim boyutuna çeviren bir input katmanı kurar. 
# Sonrasında num_blocks kadar residual blok ve en sonunda tek değer üreten output katmanı ekler.
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_blocks: int = 3, dropout_rate: float = 0.2):
        """
        Initialize the ResNet model.

        Args:
            input_dim: Number of input features
            hidden_dim: Dimension of hidden layers
            num_blocks: Number of residual blocks
            dropout_rate: Dropout probability
        """
        super(PersonnelResNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
#Girişi input katmandan geçirir ve ardından residual blokları sırayla uygular. 
# Son olarak output katmanından geçirip tek değerlik tahmini döndürür.
    def forward(self, x):
        """Forward pass through the ResNet."""
        x = self.input_layer(x)

        for block in self.res_blocks:
            x = block(x)

        return self.output_layer(x).squeeze(-1)

#Self-attention mantığıyla girişteki özelliklerin birbirine olan etkisini öğrenerek 
# daha önemli özellikleri daha fazla vurgular. Amaç, modelin “hangi özellik daha önemli” 
# bilgisini otomatik öğrenmesini sağlamaktır.
class AttentionLayer(nn.Module):
    """
    Self-attention layer for feature weighting.

    Learns to weight different input features based on their
    importance for the prediction task.
    """
#Query, Key ve Value dönüşümlerini yapmak için 3 adet lineer katman kurar.
#  Ayrıca attention skorlarının ölçeklenmesi için scale = sqrt(dim) değerini hesaplar.
    def __init__(self, dim: int):
        """
        Initialize attention layer.

        Args:
            dim: Feature dimension
        """
        super(AttentionLayer, self).__init__()

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
#Giriş tensörünü attention hesaplamasına uygun hale getirip Q(Hangi özellikler işe yarıyor?)-K(Bu özellikler ne kadar alakal)-V(Asıl taşınacak bilgi) üretir, 
# sonra attention ağırlıklarıyla V üzerinde ağırlıklı ortalama alır. Sonuç olarak girişin “önem ağırlıklı” yeni temsili döndürülür.
#Attention, “hangi bilgi önemliyse onun katkısını artırarak” girişten daha akıllı bir çıktı üretir.
    def forward(self, x):
        """
        Apply self-attention.

        Args:
            x: Input tensor of shape (batch_size, dim)

        Returns:
            Attention-weighted output
        """
        # For 1D input, we treat each feature as a "token"
        # Reshape to (batch, 1, dim) for attention computation
        x = x.unsqueeze(1)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        return out.squeeze(1)

#Önce özellikleri embedding ile daha güçlü bir temsile dönüştürür, ardından attention ile önemli özellikleri seçer 
# ve MLP ile tahmini yapar. Bu model, özellikle çok özellikli verilerde daha iyi performans gösterebilir.
class PersonnelAttentionNet(nn.Module):
    """
    Neural Network with Attention for personnel norm prediction.

    Uses attention mechanism to learn feature importance weights,
    followed by MLP layers for final prediction.

    Architecture:
        Input -> Attention -> MLP -> Output
    """
#Giriş özelliklerini hidden_dim boyutuna taşıyan embedding katmanı oluşturur ve attention katmanı ekler. 
# Sonrasında attention çıktısını işleyen küçük bir MLP kurup tek değerlik output üretir.
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 dropout_rate: float = 0.3):
        """
        Initialize the Attention Network.

        Args:
            input_dim: Number of input features
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout probability
        """
        super(PersonnelAttentionNet, self).__init__()

        self.input_dim = input_dim

        # Feature embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Attention layer
        self.attention = AttentionLayer(hidden_dim)

        # MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1)
        )
#Önce embedding uygulanır, sonra attention çalıştırılır ve attention çıktısı MLP’ye verilir. 
# Sonuç olarak regresyon tahmini üretilip tek boyuta indirgenir.
    def forward(self, x):
        """Forward pass through the attention network."""
        x = self.embedding(x)
        x = self.attention(x)
        return self.mlp(x).squeeze(-1)

#Model adını (mlp, resnet, attention) alıp ilgili sınıftan doğru modeli üretir ve döndürür. 
# Böylece kodun farklı modellerle kolayca çalıştırılabilir hale gelir.
def get_model(model_name: str, input_dim: int, **kwargs):
    """
    Factory function to create models by name.

    Args:
        model_name: One of 'mlp', 'resnet', 'attention'
        input_dim: Number of input features
        **kwargs: Additional model parameters

    Returns:
        Initialized model
    """
    models = {
        'mlp': PersonnelMLP,
        'resnet': PersonnelResNet,
        'attention': PersonnelAttentionNet
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(models.keys())}")

    return models[model_name.lower()](input_dim, **kwargs)

#Modeldeki eğitilebilir (requires_grad=True) parametreleri sayar ve toplam parametre sayısını döndürür. 
# Bu, modelin büyüklüğünü ve karmaşıklığını anlamanı sağlar.
def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Dosya direkt çalıştırılırsa tüm modelleri test eder ve giriş/çıkış 
# şekillerini ve parametre sayılarını yazdırır. 
# Böylece mimarilerin doğru çalıştığını hızlıca kontrol edebilirsin.
if __name__ == "__main__":
    # Test models
    print("Testing model architectures...")

    input_dim = 14
    batch_size = 32

    # Create sample input
    x = torch.randn(batch_size, input_dim)

    # Test MLP
    print("\n" + "="*50)
    print("MLP Model")
    print("="*50)
    mlp = PersonnelMLP(input_dim)
    out = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(mlp):,}")

    # Test ResNet
    print("\n" + "="*50)
    print("ResNet Model")
    print("="*50)
    resnet = PersonnelResNet(input_dim)
    out = resnet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(resnet):,}")

    # Test Attention Net
    print("\n" + "="*50)
    print("Attention Network")
    print("="*50)
    attn_net = PersonnelAttentionNet(input_dim)
    out = attn_net(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(attn_net):,}")
