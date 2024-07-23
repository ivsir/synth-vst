// The encrypted path as a JSON array of ASCII values
const encryptedPathAscii = [50,51,49,56,48,97,56,51,55,57,57,56,54,53,101,57,57,100,52,52,48,50,52,55,50,102,54,100,97,99,98]

// Convert the ASCII values to characters and join them into a string
const encryptedPath = encryptedPathAscii.map(code => String.fromCharCode(code)).join('');

console.log(`Decrypted path: ${encryptedPath}`);
