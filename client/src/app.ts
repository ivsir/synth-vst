import * as Tone from "tone";
import { WebMidi } from "webmidi";
import * as musicData from "../chords/chords.json";

import "./sass/style.scss";

import Index from "./components/index.js";

import midiMapper from "./functions/midiMapper.js";
import keyboardMapper from "./functions/keyboardMapper.js";

const root = document.getElementById("root");
if (root) {
  root.innerHTML = Index();
}

// make and start a 440hz sine tone
const synth = new Tone.PolySynth().toDestination();

// <---------------------------------------------------->

const piano = document.getElementById("piano");
if (piano) {
  for (let note = 21; note <= 108; note++) {
    const noteName = midiMapper(note);
    if (noteName) {
      const keyElement = document.createElement("div");
      keyElement.classList.add("key");
      keyElement.classList.add(noteName.includes("b") ? "black" : "white");
      keyElement.dataset.note = noteName; // Store note name for reference
      piano.appendChild(keyElement);

      // Initialize mouse events for the key element
      mouseEvents(keyElement, noteName);
      // keyboardEvents(keyElement, noteName);
      keyboardEvents(keyElement, noteName);
    }
  }
}

// <---------------------------------------------------->

type ChordData = {
  name: string;
  notes: string[];
  intervals: string[];
  midiKeys: number[];
};
type MusicData = {
  [key: string]: {
    [chord: string]: ChordData;
  };
};

function removeNumbers(note: string): string {
  return note.replace(/[0-9]/g, "");
}

function logMatchingData(jsonData: MusicData, targetNote: string) {
  const normalizedTargetNote = removeNumbers(targetNote);
  console.log("normalizedTargetNote", normalizedTargetNote);

  // Normalize the keys in jsonData to ignore numbers as well
  const normalizedKeys = Object.keys(jsonData).map((key) => removeNumbers(key));

  // Find the index of the normalized target note in the normalized keys
  const targetIndex = normalizedKeys.indexOf(normalizedTargetNote);

  // Get the container to display the chords
  const container = document.getElementById("chordContainer");
  if (!container) return;

  // Clear previous contents
  container.innerHTML = "";

  if (targetIndex !== -1) {
    const originalKey = Object.keys(jsonData)[targetIndex];
    const chords = jsonData[originalKey];

    // Create a header for the key
    const keyHeader = document.createElement("h2");
    keyHeader.textContent = `Key: ${normalizedTargetNote}`;
    container.appendChild(keyHeader);

    // Create a list of chords
    const chordList = document.createElement("ul");
    for (const chordName in chords) {
      const chord = chords[chordName];

      // Create a list item for each chord
      const chordItem = document.createElement("li");
      chordItem.textContent = `Chord Name: ${normalizedTargetNote}-${chordName}, Notes: ${chord.notes.join(
        ", "
      )}, Intervals: ${chord.intervals.join(
        ", "
      )}, MIDI Keys: ${chord.midiKeys.join(", ")}`;
      chordList.appendChild(chordItem);
    }

    container.appendChild(chordList);
    console.log(`key: ${normalizedTargetNote}, chords:`, chords);
  } else {
    const noMatchMessage = document.createElement('p');
    noMatchMessage.textContent = `No matching key found for ${normalizedTargetNote}`;
    console.log(`No matching key found for ${normalizedTargetNote}`);
  }
}

// <---------------------------------------------------->

let playChord = []

function playNote(note: string) {
  const now = Tone.now();
  logMatchingData(musicData as MusicData, note);
  
  synth.triggerAttack(note, now);
}

function stopNote(note: string) {
  const now = synth.now();
  synth.triggerRelease(note, now);
}

// <---------------------------------------------------->

function mouseEvents(element: HTMLElement, note: string) {
  const playNoteHandler = () => {
    element.classList.add("active");
    playNote(note);
  };
  const stopNoteHandler = () => {
    element.classList.remove("active");
    stopNote(note);
  };
  const handleMouseEnter = (event: MouseEvent) => {
    if ((event.buttons & 1) === 1) {
      element.classList.add("active");
      playNoteHandler();
    }
  };

  element.addEventListener("mousedown", playNoteHandler);
  element.addEventListener("mouseup", stopNoteHandler);
  element.addEventListener("mouseleave", stopNoteHandler);
  element.addEventListener("mouseenter", handleMouseEnter);
}

// <---------------------------------------------------->

function keyboardEvents(element: HTMLElement, note: string) {
  let baseOffset = 0;

  const playNoteHandler = (note: string) => {
    element.classList.add("active");
    playNote(note);
  };
  const stopNoteHandler = (note: string) => {
    element.classList.remove("active");
    stopNote(note);
  };

  const keydownHandler = (event: KeyboardEvent) => {
    if (event.code === "KeyZ") {
      baseOffset -= 12; // Move down an octave
    } else if (event.code === "KeyX") {
      baseOffset += 12; // Move up an octave
    } else {
      const keyIndex = keyboardMapper(event.code);
      if (keyIndex !== undefined) {
        const newIndex = keyIndex + baseOffset;
        if (newIndex >= 21 && newIndex <= 108) {
          const noteName = midiMapper(newIndex);
          if (noteName === note) {
            element.classList.add("active");
            playNoteHandler(noteName);
          }
        }
      }
    }
  };

  const keyupHandler = (event: KeyboardEvent) => {
    const keyIndex = keyboardMapper(event.code);
    if (keyIndex !== undefined) {
      const newIndex = keyIndex + baseOffset;
      if (newIndex >= 21 && newIndex <= 108) {
        const noteName = midiMapper(newIndex);
        if (noteName === note) {
          element.classList.remove("active");
          stopNoteHandler(noteName);
        }
      }
    }
  };

  document.addEventListener("keydown", keydownHandler);
  document.addEventListener("keyup", keyupHandler);

  // Clean up event listeners on component unmount or similar
  return () => {
    document.removeEventListener("keydown", keydownHandler);
    document.removeEventListener("keyup", keyupHandler);
  };
}
