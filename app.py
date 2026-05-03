import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

if not shutil.which("ffmpeg"):
    local_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin")
    if os.path.isdir(local_ffmpeg):
        os.environ["PATH"] += os.pathsep + local_ffmpeg

from flask import Flask, request, jsonify, render_template
import re, yt_dlp, librosa, numpy as np, joblib, uuid
import onnxruntime as ort
from skimage.transform import resize

app = Flask(__name__)

# 🔥 modèle
session = ort.InferenceSession("modele_cnn.onnx")
le = joblib.load("label_encoder_cnn.pkl")

YOUTUBE_RE = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11}"
)

GENRE_EMOJI = {
    "blues": "🎸", "classical": "🎻", "country": "🤠",
    "disco": "🕺",  "hiphop": "🎤",   "jazz": "🎷",
    "metal": "🤘",  "pop": "🎵",      "reggae": "🌿",
    "rock": "⚡",
}

# =========================
# 🔥 FEATURE EXTRACTION AMÉLIORÉE
# =========================
def extract_features(filepath: str) -> np.ndarray:
    # 🔥 ignorer intro (très important)
    audio, sr = librosa.load(filepath, sr=22050, mono=True, offset=15, duration=25)

    # 🔥 mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 🔥 normalisation robuste
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # 🔥 resize stable
    mel_resized = resize(mel_db, (128, 128), anti_aliasing=True)

    # 🔥 smoothing (IMPORTANT)
    mel_resized = np.clip(mel_resized, -3, 3)

    return mel_resized.astype(np.float32).reshape(1, 128, 128, 1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URL manquante"}), 400
    if not YOUTUBE_RE.search(url):
        return jsonify({"error": "URL YouTube invalide"}), 400

    tmp_id = uuid.uuid4().hex

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"tmp_{tmp_id}.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        # 🔥 download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "")

        files = [f for f in os.listdir(".") if f.startswith(f"tmp_{tmp_id}")]
        if not files:
            return jsonify({"error": "Audio non trouvé"}), 500

        # 🔥 features
        feats = extract_features(files[0])

        # 🔥 prediction
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: feats})[0][0]

        # 🔥 TOP 3 (très important pour éviter erreurs absurdes)
        top3_idx = np.argsort(pred)[::-1][:3]

        top3 = []
        for i in top3_idx:
            genre = le.inverse_transform([i])[0]
            top3.append({
                "genre": genre,
                "emoji": GENRE_EMOJI.get(genre, "🎵"),
                "confidence": round(float(pred[i]) * 100, 1),
            })

        return jsonify({
            "genre": top3[0]["genre"],
            "emoji": top3[0]["emoji"],
            "confidence": f"{top3[0]['confidence']:.1f}%",
            "top3": top3,
            "title": title,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # 🔥 clean fichiers
        for f in os.listdir("."):
            if f.startswith(f"tmp_{tmp_id}"):
                try:
                    os.remove(f)
                except:
                    pass


if __name__ == "__main__":
    app.run(debug=True, port=5001)