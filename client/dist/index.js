// src/index.ts
import * as Tone from 'tone';
const synth = new Tone.Synth().toDestination();
Tone.start().then(() => {
    synth.triggerAttackRelease('C4', '8n');
    console.log('Sound played');
});
