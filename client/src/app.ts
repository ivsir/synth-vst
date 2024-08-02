import * as Tone from "tone";
import { WebMidi } from "webmidi";

import "./sass/style.scss";

import Index from "./components/index.js";

import midiMapper from "./functions/midiMapper.js";
import keyboardMapper from "./functions/keyboardMapper.js";

import axios from "axios";

const root = document.getElementById("root");
if (root) {
  root.innerHTML = Index();
}

// make and start a 440hz sine tone
const synth = new Tone.PolySynth().toDestination();

// <---------------------------- MIDI ACCESS ------------------------>
WebMidi.enable({
  callback: (err: any) => {
    if (err) {
      console.error("WebMidi could not be enabled.", err);
    } else {
      console.log("WebMidi enabled!");

      const midiInputSelect = document.getElementById(
        "midiInputSelect"
      ) as HTMLSelectElement;
      if (midiInputSelect) {
        // Populate the select field with available MIDI input devices
        WebMidi.inputs.forEach((input) => {
          const option = document.createElement("option");
          option.value = input.id;
          option.text = input.name;
          midiInputSelect.appendChild(option);
        });

        // Add event listener for MIDI input selection
        midiInputSelect.addEventListener("change", (event) => {
          const selectedId = (event.target as HTMLSelectElement).value;
          const selectedInput = WebMidi.getInputById(selectedId);
          if (selectedInput) {
            setupMIDIInput(selectedInput);
          }
        });
      }
    }
  },
});

// Function to setup MIDI input event listeners
function setupMIDIInput(input: any) {
  // Remove event listeners from previous input if any
  WebMidi.inputs.forEach((input) => {
    input.removeListener("noteon");
    input.removeListener("noteoff");
  });

  // Add event listeners for the selected input
  input.addListener("noteon", "all", (e: any) => {
    console.log("e.note", e.note);
    console.log(
      `Received 'noteon' message (${e.note.name}${e.note._accidental || ""}${
        e.note.octave
      }).`
    );

    // Construct the note string with or without accidental
    const noteString = `${e.note.name}${e.note._accidental || ""}${
      e.note.octave
    }`;
    const element = document.querySelector(
      `[data-note="${noteString}"]`
    ) as HTMLElement;
    playNoteFromMidi(e.note.number, element);
  });

  input.addListener("noteoff", "all", (e: any) => {
    console.log(
      `Received 'noteoff' message (${e.note.name}${e.note._accidental || ""}${
        e.note.octave
      }).`
    );

    // Construct the note string with or without accidental
    const noteString = `${e.note.name}${e.note._accidental || ""}${
      e.note.octave
    }`;
    const element = document.querySelector(
      `[data-note="${noteString}"]`
    ) as HTMLElement;
    stopNoteFromMidi(e.note.number, element);
  });
}

// <---------------------------------------------------->

const piano = document.getElementById("piano");
if (piano) {
  for (let note = 21; note <= 108; note++) {
    const noteName = midiMapper(note);
    if (noteName) {
      const noteMidi = note;
      const keyElement = document.createElement("div");
      keyElement.classList.add("key");
      keyElement.classList.add(noteName.includes("#") ? "black" : "white");
      keyElement.dataset.note = noteName; // Store note name for reference
      piano.appendChild(keyElement);

      // Initialize mouse events for the key element
      mouseEvents(keyElement, noteName, noteMidi);
      // keyboardEvents(keyElement, noteName);
      keyboardEvents(keyElement, noteName);
    }
  }
}

// <---------------------------------------------------->

interface MidiEvent {
  note: string;
  midi: number; // MIDI number
  timestamp: number; // Time at which the event occurred
}

let playChord: MidiEvent[] = []; // Array to store MIDI numbers with timestamps
const TIME_WINDOW = 0.18;

async function addNoteToPlayChord(note: string, midi: number) {
  const now = Tone.now(); // Current time in seconds

  // Add the new note to the playChord array
  playChord.push({ note, midi, timestamp: now });
  console.log(playChord);
  // Remove events older than TIME_WINDOW seconds
  playChord = playChord.filter((event) => now - event.timestamp <= TIME_WINDOW);
  // Send the most recent playChord to the backend
  try {
    const response = await axios.post("http://localhost:5001/predict_chord", {
      // midi_keys: playChord.map((event) => event.midi),
      midi_keys: playChord.map((event) => event.note),
    });
    // Get the container to display the chords
    const container = document.getElementById("chordContainer");
    if (!container) return;

    container.innerHTML = "";
    // Display the predicted chord name on the frontend
    const predictedChord = response.data.chord_name;

    // Create a header for the key
    const keyHeader = document.createElement("li");
    keyHeader.textContent = `Chord: ${predictedChord}`;
    container.appendChild(keyHeader);
    // You can update your UI with the predicted chord here
  } catch (error) {
    console.error("Error predicting chord:", error);
  }
}

function playNote(note: string, midi: number) {
  const now = Tone.now();
  // Add the note to the playChord array
  addNoteToPlayChord(note, midi);

  synth.triggerAttack(note, now);
}

function stopNote(note: string, midi: number) {
  const now = Tone.now();
  synth.triggerRelease(note, now);
}

function playNoteFromMidi(midi: number, element: HTMLElement) {
  const note = midiMapper(midi);
  element.classList.add("active");
  if (note) {
    playNote(note, midi);
  }
}

function stopNoteFromMidi(midi: number, element: HTMLElement) {
  const note = midiMapper(midi);
  element.classList.remove("active");
  if (note) {
    stopNote(note, midi);
  }
}

// <---------------------------------------------------->

function mouseEvents(element: HTMLElement, note: string, midi: number) {
  const playNoteHandler = () => {
    element.classList.add("active");
    playNote(note, midi);
  };
  const stopNoteHandler = () => {
    element.classList.remove("active");
    stopNote(note, midi);
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

  const playNoteHandler = (note: string, midi: number) => {
    element.classList.add("active");
    playNote(note, midi);
  };
  const stopNoteHandler = (note: string, midi: number) => {
    element.classList.remove("active");
    stopNote(note, midi);
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
          const midiNumber = newIndex;
          if (noteName === note) {
            element.classList.add("active");
            playNoteHandler(noteName, midiNumber);
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
        const midiNumber = newIndex;
        if (noteName === note) {
          element.classList.remove("active");
          stopNoteHandler(noteName, midiNumber);
        }
      }
    }
  };

  document.addEventListener("keydown", keydownHandler);
  document.addEventListener("keyup", keyupHandler);
}
