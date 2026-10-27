import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

BASE_DIR = Path(__file__).parent

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

VISION_MODEL = os.environ.get("VISION_MODEL", "claude-sonnet-5")
CONDUTA_MODEL = os.environ.get("CONDUTA_MODEL", "claude-sonnet-5")  # conduta = output de maior risco clínico
LAUDO_POLISH_MODEL = "claude-sonnet-4-6"
GT_PARSE_MODEL = "claude-haiku-4-5-20251001"

# Raciocínio estendido na leitura da imagem. Piloto (out/2026): Sonnet 5 + thinking
# bateu Haiku sem thinking e empatou/superou Opus 4.8 por ~1/3 do custo do Opus.
# Com thinking ligado, o tool_choice NÃO pode ser forçado (limitação da API), então
# o extractor usa tool_choice=auto + instrução forte + desembrulho de wrapper.
VISION_THINKING = os.environ.get("VISION_THINKING", "1") not in ("0", "false", "False", "")
VISION_EFFORT = os.environ.get("VISION_EFFORT", "high")   # low|medium|high|xhigh|max

DB_PATH = BASE_DIR / "data" / "ecg.db"
EVAL_IMAGES_DIR = (BASE_DIR.parent / "imagens" / "Imagens_ECG").resolve()
EVAL_XLSX_PATH = (BASE_DIR.parent / "dataset" / "ECG_em_10.xlsx").resolve()
PROD_IMAGES_DIR = BASE_DIR / "data" / "images" / "production"

PIPELINE_VERSION = "v2.4.0-sonnet5-thinking"

# Few-shot dinâmico na conduta: injeta exemplos validados (voz da médica + GT v1)
# pra captar o estilo de escrita dela. Desliga com FEWSHOT_CONDUTA=0.
FEWSHOT_CONDUTA = os.environ.get("FEWSHOT_CONDUTA", "1") not in ("0", "false", "False", "")
# Few-shot dinâmico no LAUDO: injeta laudos validados pela médica como turnos
# user/assistant pra o polish capturar a voz dela (e não só limpar o template).
# Cai pro polish constrangido se a recuperação não trouxer exemplos.
FEWSHOT_LAUDO = os.environ.get("FEWSHOT_LAUDO", "1") not in ("0", "false", "False", "")
FEWSHOT_K = int(os.environ.get("FEWSHOT_K", "3"))

# Pré-processamento clássico (OpenCV) antes da vision. Cada estágio pode ser
# desligado individualmente em A/B; falha de qualquer um devolve a imagem
# original (preferimos imagem ruim a quebra de pipeline).
PREPROCESS_VISION = os.environ.get("PREPROCESS_VISION", "1") not in ("0", "false", "False", "")
PREPROCESS_DESKEW = os.environ.get("PREPROCESS_DESKEW", "1") not in ("0", "false", "False", "")
PREPROCESS_CROP_PAPER = os.environ.get("PREPROCESS_CROP_PAPER", "1") not in ("0", "false", "False", "")
PREPROCESS_CLAHE = os.environ.get("PREPROCESS_CLAHE", "1") not in ("0", "false", "False", "")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GRUPO_ECG_ID = int(os.environ.get("GRUPO_ECG_ID", "0") or 0)
GRUPO_REVISAO_ID = int(os.environ.get("GRUPO_REVISAO_ID", "0") or 0)
_allowed = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
TELEGRAM_ALLOWED_CHAT_IDS = {int(x) for x in _allowed.split(",") if x.strip()} if _allowed else set()

PROD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
