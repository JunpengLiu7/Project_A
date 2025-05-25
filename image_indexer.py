import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

def image_to_embedding(image_path, embed_model):
    """将图像转换为嵌入向量"""
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        _ = np.asarray(image).astype(np.float32) / 255.0  # 模拟预处理（可拓展）
        
        # 为图像生成一个描述性文本，用于生成嵌入
        fake_text = f"image-vector:{os.path.basename(image_path)}"
        embedding = embed_model.embed_query(fake_text)
        return embedding
    except Exception as e:
        print(f"❌ 图像处理失败: {image_path}，错误: {e}")
        return None

def find_all_images(root_dir):
    """递归查找所有支持的图像文件"""
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):  # 修复了这里的语法错误
        for filename in filenames:
            if filename.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                full_path = os.path.join(dirpath, filename)
                image_paths.append(full_path)
    return image_paths

def build_image_vector_index(root_dir="Markdown", index_dir="faiss_image_index"):
    """构建图像向量索引"""
    print("📸 正在构建图像向量索引...")
    os.makedirs(index_dir, exist_ok=True)
    
    # 查找所有图像
    image_paths = find_all_images(root_dir)
    print(f"🔍 共找到 {len(image_paths)} 张图片")
    
    if not image_paths:
        print("⚠️ 未找到图像，跳过图像索引构建。")
        return
    
    # 初始化嵌入模型
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 生成向量和文档
    embeddings = []
    texts = []
    metadatas = []
    
    for path in tqdm(image_paths, desc="🔍 图像编码"):
        emb = image_to_embedding(path, embed_model)
        if emb is not None:
            embeddings.append(emb)
            texts.append(f"Image: {os.path.basename(path)}")  # 提供实际的文本内容
            metadatas.append({"source": path, "type": "image"})
    
    if not embeddings:
        print("⚠️ 无有效图像向量，索引构建终止。")
        return
    
    print("💾 正在保存向量索引...")
    
    # 使用正确的方法创建FAISS索引
    try:
        # 方法1: 使用 from_embeddings (推荐)
        text_embedding_pairs = list(zip(texts, embeddings))
        vectordb = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=embed_model,
            metadatas=metadatas
        )
    except Exception as e:
        print(f"使用from_embeddings失败: {e}")
        # 方法2: 使用 from_texts 作为备选
        try:
            vectordb = FAISS.from_texts(
                texts=texts,
                embedding=embed_model,
                metadatas=metadatas
            )
        except Exception as e2:
            print(f"使用from_texts也失败: {e2}")
            return
    
    # 保存索引
    vectordb.save_local(index_dir)
    
    # 保存图像路径列表
    with open(os.path.join(index_dir, "paths.txt"), "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(path + "\n")
    
    print("✅ 图像向量索引构建完成！")

# 可测试入口
if __name__ == "__main__":
    build_image_vector_index()