export default function keyboardMapper(key: string) {
  const keyMap: { [key: string]: number } = {
      KeyA: 60,
      KeyW: 61,
      KeyS: 62,
      KeyE: 63,
      KeyD: 64,
      KeyF: 65,
      KeyT: 66,
      KeyG: 67,
      KeyY: 68,
      KeyH: 69,
      KeyU: 70,
      KeyJ: 71,
      KeyK: 72,
      KeyO: 73,
      KeyL: 74,
      KeyP: 75,
  };

  if (key in keyMap) {
      return keyMap[key];
  }
}