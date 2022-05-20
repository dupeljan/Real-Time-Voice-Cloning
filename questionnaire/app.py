import sqlite3
import os

from pathlib import Path
from flask import (
    Flask,
    render_template,
    send_file,
    request,
    redirect,
    url_for,
    session
    )

app = Flask(__name__)
if 'FLASK_SECRET_KEY' not in os.environ:
    raise RuntimeError('\'FLASK_SECRET_KEY\' env var is not specified')
app.secret_key = bytes(os.environ['FLASK_SECRET_KEY'], 'utf-8')

os.environ['VOICES_PATH'] = '/home/dupeljan/Projects/master_diploma/generated_voices' 

gen_type_map = {
    'ref': 'ref_voices',
    'current_without_heuristics': 'default_output',
    'current_with_heuristics': 'mixed_output',
    'previous': 'output_prev',
}

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

DATA_PATH = Path(os.environ['VOICES_PATH']) 
SPEAKERS = [speaker.name for speaker in (DATA_PATH / gen_type_map['ref']).glob('*')]

@app.route('/start', methods=('GET', 'POST'))
def gen_index():
    if request.method == 'POST':
        session['user'] = request.form['name'] 
    if 'user' in session:
        return redirect(f'/eval_voice_cloning/{SPEAKERS[0]}') 
    return render_template('start.html')


@app.route('/eval_voice_cloning/<speaker>', methods=('GET', 'POST'))
def gen_speaker(speaker):
    def redirect_to_next_speaker():
        speaker_idx = SPEAKERS.index(speaker)
        if speaker_idx + 1 == len(SPEAKERS):
            return redirect(url_for('finish'))
        return redirect(f'/eval_voice_cloning/{SPEAKERS[speaker_idx + 1]}') 

    if request.method == 'POST':

        
        conn = get_db_connection()
        for k in request.form:
            gen_type, _, audio_idx, mode = k.split('-')
            conn.execute('INSERT INTO records (user, speaker, gen_type, audio_idx, mode) VALUES (?, ?, ?, ?, ?)',
                         (session['user'], speaker, gen_type, audio_idx, mode))
        conn.commit()
        conn.close()
        return redirect_to_next_speaker()
    else:
        if 'user' not in session:
            return f'Please login! <a href="{url_for("gen_index")}">Login</a>'

        if speaker not in SPEAKERS:
            return f'Speaker {speaker} is not exist'

        conn = get_db_connection()
        rows = conn.execute('SELECT id FROM records WHERE SPEAKER=? AND USER=?', (speaker, session['user'])).fetchall()
        if rows:
            return redirect_to_next_speaker()
        speaker_idx = SPEAKERS.index(speaker)
        payload = [] 
        lines = open(DATA_PATH / 'default_texts.txt', 'r').readlines()
        for idx, line in enumerate(lines):
            item = {
                'gen_types': [gen_type_map[key] for key in ['previous', 
                                                            'current_without_heuristics',
                                                            'current_with_heuristics']],
                'speaker': speaker, 
                'audio_idx': idx,
                'text': line,
            }
            payload.append(item)
    
        return render_template('index.html', payload=payload, speaker=speaker, ref_type=gen_type_map['ref'],
                               page_num=speaker_idx + 1, pages_total=len(SPEAKERS)) 


@app.route('/finish')
def finish():
    if 'user' not in session:
        return redirect(url_for('gen_index'))
    return f'Thanks for your participation, {session["user"]}!'


@app.route('/eval_voice_cloning/audio/<gen_type>/<speaker>/<int:audio_idx>')
def send_audio(gen_type, speaker, audio_idx):
    path_to_audio_file = Path(os.environ['VOICES_PATH']) / gen_type / speaker / f'{audio_idx}.wav'
    return send_file(
       path_to_audio_file, 
       mimetype="audio/wav", 
       as_attachment=True, 
       attachment_filename=f"{gen_type}_{speaker}_{audio_idx}.wav")
