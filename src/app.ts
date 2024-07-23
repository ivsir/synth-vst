import * as Tone from "tone";
import * as Nexus from "nexusui2";
import { WebMidi } from "webmidi";

import "./sass/style.scss";

import Index from "./components/index.js"

// 
import midiMapper from "./functions/midiMapper.js"

const root = document.getElementById("root");
if (root) {
  root.innerHTML = Index();
}

// make and start a 440hz sine tone
const synth = new Tone.Synth().toDestination()

const piano = document.getElementById('piano');

function playNote(note: string){
  // <-------------------------------- Gets Time of Current Event-------------------------------->
  const now = Tone.now();
  // <-------------------------------- Gets Time of Current Event-------------------------------->
  synth.triggerAttack(note, now)
}

function stopNote(note: string){
  const now = Tone.now();
  synth.triggerRelease(now)
}

function mouseEvents(keyElement: HTMLElement, note: string) {
  keyElement.addEventListener("mousedown", () => {
    keyElement.classList.add('active');
    playNote(note);
  });

  keyElement.addEventListener("mouseup", () => {
    keyElement.classList.remove('active');
    stopNote(note);
  });

  keyElement.addEventListener('mouseleave', () => {
    if (keyElement.classList.contains('active')) {
      keyElement.classList.remove('active');
      stopNote(note);
    }
  });

  keyElement.addEventListener("mouseenter", (event) => {
    if ((event.buttons & 1) === 1) { // Check if the left mouse button is held down
      keyElement.classList.add('active');
      playNote(note);
    }
  });
}

if (piano) {
  for (let note = 21; note <= 108; note++) {
    const noteName = midiMapper(note);
    if (noteName) {
      const keyElement = document.createElement('div');
      keyElement.classList.add('key');
      keyElement.classList.add(noteName.includes('b') ? 'black' : 'white');
      keyElement.dataset.note = noteName; // Store note name for reference
      piano.appendChild(keyElement);

      // Initialize mouse events for the key element
      mouseEvents(keyElement, noteName);
    }
  }
}