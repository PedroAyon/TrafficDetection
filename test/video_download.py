import yt_dlp

url = "https://www.youtube.com/watch?v=wqctLW0Hb_0"
url = "https://www.youtube.com/watch?v=lfYYtQ1Ah04"

ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': 'video2.%(ext)s',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
