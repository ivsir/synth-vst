import PianoKeyboard from "./panels/pianoKeyboard";
import ChordProgressions from "./panels/chords";
import MIDIInputSelect from "./panels/midiDevice";

export default function index() {
  return /*html*/ `
  <div class="synthContainer">
      ${MIDIInputSelect()}
      ${PianoKeyboard()}
      ${ChordProgressions()}
  </div>
  `;
}
