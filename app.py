import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ── ffmpeg : cherche dans PATH, puis dans le dossier courant ──────────────────
import shutil

if not shutil.which("ffmpeg"):
    os.environ["PATH"] += os.pathsep + r"C:\Users\aouar\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

from flask import Flask, request, jsonify, render_template
import re, yt_dlp, librosa, numpy as np, joblib, uuid
import onnxruntime as ort

app = Flask(__name__)

# ── Chargement du modèle ──────────────────────────────────────────────────────
session = ort.InferenceSession("modele_cnn.onnx")
le      = joblib.load("label_encoder_cnn.pkl")

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

def extract_features(filepath: str) -> np.ndarray:
    """Mel-spectrogram → tenseur (1, 128, 128, 1).
    Normalisation identique à l'entraînement : mean/std par image."""
    audio, sr = librosa.load(filepath, duration=30, mono=True)
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # ✅ même normalisation qu'à l'entraînement
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # redimensionner à 128×128 avec numpy (sans PIL pour éviter la conversion float→uint8)
    from skimage.transform import resize as sk_resize
    mel_r = sk_resize(mel_db, (128, 128), anti_aliasing=True).astype(np.float32)
    return mel_r.reshape(1, 128, 128, 1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url  = data.get("url", "").strip()

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
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "")

        fichiers = [f for f in os.listdir(".") if f.startswith(f"tmp_{tmp_id}")]
        if not fichiers:
            return jsonify({"error": "Fichier audio non trouvé après téléchargement"}), 500

        feats      = extract_features(fichiers[0])
        input_name = session.get_inputs()[0].name
        pred       = session.run(None, {input_name: feats})[0][0]  # shape (n_classes,)

        # Top-3 genres
        top3_idx   = np.argsort(pred)[::-1][:3]
        top3 = [
            {
                "genre": le.inverse_transform([i])[0],
                "emoji": GENRE_EMOJI.get(le.inverse_transform([i])[0], "🎵"),
                "confidence": round(float(pred[i]) * 100, 1),
            }
            for i in top3_idx
        ]

        return jsonify({
            "genre":      top3[0]["genre"],
            "emoji":      top3[0]["emoji"],
            "confidence": f"{top3[0]['confidence']:.1f}%",
            "top3":       top3,
            "title":      title,
        })

    except yt_dlp.utils.DownloadError as e:
        return jsonify({"error": f"Téléchargement impossible : {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for f in os.listdir("."):
            if f.startswith(f"tmp_{tmp_id}"):
                try:
                    os.remove(f)
                except OSError:
                    pass


if __name__ == "__main__":
    app.run(debug=True)