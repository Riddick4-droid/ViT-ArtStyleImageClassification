import torch
import torch.nn as nn
from src.logger import get_logger
from src.exceptions import ModelBuildError

logger = get_logger(__name__)

#patch embeddings
class PatchEmbedding(nn.Module):
    """
    Split image into fixed-size patches and project to embedding dimension.
    Uses a Conv2d layer for efficient computation.
    """
    def __init__(self, img_size:int, patch_size:int, in_channels: int, embed_dim:int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ModelBuildError(f"Image size {img_size} must be divisible by patch size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels,embed_dim, kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        #image comes in as (B,C,H,W), becomes (B,embed_dim, H/P, W/P) where p=patch_size
        x = self.proj(x)
        x = x.flatten(2) #(B,embed_dim,N)
        x = x.transpose(1,2) #(B,N,embed_dim)
        return x
    
#positional embeddings + cls token
class PositionalEmbedding(nn.Module):
    """
    Adds a learnable CLS token and positional embeddings to the patch sequence.
    If use_cls_token=False, only positional embeddings are added (for pooling).
    """
    def __init__(self, num_patches:int, embed_dim:int, use_cls_token:bool=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        if self.use_cls_token:#
            self.cls_token = nn.Parameter(torch.randn(1,1, embed_dim))
            self.pos_embed = nn.Parameter(torch.randn(1,num_patches+1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        #imitialize the weights with truncated normal (std=0.02 as original vit)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            nn.init.trunc_normal_(self.cls_token,std=0.02)

    def forward(self,x):
        #x comes from the patch embed (B,N,embed_dim) patch tokens
        B = x.shape[0]
        if self.use_cls_token:
            cls_token = self.cls_token.expand(B,-1,1)
            x = torch.cat((cls_token,x),dim=1)
        x = x + self.pos_embed
        return x

#transformer encoder block
class EncoderBlock(nn.Module):
    """One transformer block: Pre-norm MHA + MLP with residual connections."""
    def __init__(self, embed_dim:int, num_heads:int, mlp_ratio:float=4.0, dropout:float=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
             nn.Linear(embed_dim,hidden_dim),
             nn.GELU(),
             nn.Dropout(dropout),
             nn.Linear(hidden_dim,embed_dim),
             nn.Dropout(dropout)
         )
    def forward(self,x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
#transformer block (stack of blocks)
class TransformerBlock(nn.Module):
    """Stack of EncoderBlock + final LayerNorm."""
    def __init__(self, embed_dim: int, num_heads: int, depth: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio,dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self,x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
#classification head
class ClassificationHead(nn.Module):
    """final linear layer"""
    def __init__(self, embed_dim:int, num_classes:int):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self,x):
        return self.head(x)
    

#full vision transformer
class CustomViT(nn.Module):
    def __init__(self, img_size:int=224, patch_size: int=16, in_channels:int=3,num_classes:int=1000,embed_dim:int=768,
                 depth:int=12, num_heads:int=12, mlp_ratio:float=4.0, dropout:float=0.1, use_cls_token:bool=True):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim, use_cls_token)
        self.encoder = TransformerBlock(embed_dim,num_heads,depth,mlp_ratio,dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = ClassificationHead(embed_dim,num_classes)
        self.use_cls_token = use_cls_token
    def forward(self,x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        if self.use_cls_token:
            x = x [:,0] #take cls token
        else:
            x = x.mean(dim=1)
        x = self.head(x)
        return x
    
#factory for building moddel from config
def build_model(config: dict, num_classes: int) -> CustomViT:
    """Instantiate a VisionTransformer from a configuration dictionary."""
    model_cfg = config['model']
    model = CustomViT(
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size'],
        in_channels=model_cfg['in_channels'],
        num_classes=num_classes,
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        mlp_ratio=model_cfg['mlp_ratio'],
        dropout=model_cfg['dropout'],
        use_cls_token=model_cfg['use_cls_token']
    )
    logger.info(f"Built ViT: {num_classes} classes, embed_dim={model.embed_dim}, "
                f"depth={model.encoder.blocks.__len__()}, heads={model_cfg['num_heads']}")
    return model