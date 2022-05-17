import os

from pathlib import Path
from itertools import islice
from flask import Flask, render_template, send_file

app = Flask(__name__)

os.environ['VOICES_PATH'] = '/home/dupeljan/Projects/master_diploma/generated_voices' 

gen_type_map = {
    'ref': 'ref_voices',
    'current_without_heuristics': 'default_output',
    'previous': 'output_prev'
}

@app.route('/eval_voice_cloning')
def gen_index():
    data_path = Path(os.environ['VOICES_PATH']) 
    payload = [] 
    speakers = [speaker.name for speaker in (data_path / gen_type_map['ref']).glob('*')]
    lines = open(data_path / 'default_texts.txt', 'r').readlines()
    for gen_type in ['current_without_heuristics', 'previous']:
        for speaker in [speakers[1]]:
            for idx, line in enumerate(lines):
                item = {
                    'gen_type': gen_type_map[gen_type],
                    'ref_type': gen_type_map['ref'],
                    'speaker': speaker, 
                    'audio_idx': idx,
                    'text': line,
                }
                payload.append(item)
    
    return render_template('index.html', payload=payload) 


@app.route('/audio/<gen_type>/<speaker>/<int:audio_idx>')
def send_audio(gen_type, speaker, audio_idx):
    path_to_audio_file = Path(os.environ['VOICES_PATH']) / gen_type / speaker / f'{audio_idx}.wav'
    return send_file(
       path_to_audio_file, 
       mimetype="audio/wav", 
       as_attachment=True, 
       attachment_filename=f"{gen_type}_{speaker}_{audio_idx}.wav")
