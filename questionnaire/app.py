import os

from pathlib import Path
from itertools import islice
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

os.environ['VOICES_PATH'] = '/home/dupeljan/Projects/master_diploma/generated_voices' 

gen_type_map = {
    'ref': 'ref_voices',
    'current_without_heuristics': 'default_output',
    'previous': 'output_prev'
}

DATA_PATH = Path(os.environ['VOICES_PATH']) 
SPEAKERS = [speaker.name for speaker in (DATA_PATH / gen_type_map['ref']).glob('*')]

@app.route('/start')
def gen_index():
    return redirect(f'/eval_voice_cloning/{SPEAKERS[0]}') 


@app.route('/eval_voice_cloning/<speaker>', methods=('GET', 'POST'))
def gen_speaker(speaker):
    if request.method == 'POST':
        print(request.form.keys())
        return redirect(f'/eval_voice_cloning/{SPEAKERS[0]}') 
    else:
        if speaker not in SPEAKERS:
            return f'Speaker {speaker} is not exist'

        payload = [] 
        lines = open(DATA_PATH / 'default_texts.txt', 'r').readlines()
        for gen_type in ['current_without_heuristics', 'previous']:
            for idx, line in enumerate(lines):
                item = {
                    'gen_type': gen_type_map[gen_type],
                    'ref_type': gen_type_map['ref'],
                    'speaker': speaker, 
                    'audio_idx': idx,
                    'text': line,
                }
                payload.append(item)
    
        return render_template('index.html', payload=payload, speaker=speaker, ref_type=gen_type_map['ref']) 


@app.route('/eval_voice_cloning/audio/<gen_type>/<speaker>/<int:audio_idx>')
def send_audio(gen_type, speaker, audio_idx):
    path_to_audio_file = Path(os.environ['VOICES_PATH']) / gen_type / speaker / f'{audio_idx}.wav'
    return send_file(
       path_to_audio_file, 
       mimetype="audio/wav", 
       as_attachment=True, 
       attachment_filename=f"{gen_type}_{speaker}_{audio_idx}.wav")
