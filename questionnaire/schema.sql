DROP TABLE IF EXISTS records;

CREATE TABLE records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    speaker TEXT NOT NULL,
    gen_type TEXT NOT NULL,
    audio_idx INTEGER NOT NULL,
    mode TEXT NOT NULL
);