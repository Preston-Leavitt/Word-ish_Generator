const { URL } = require('url');

// Helper functions for safe input handling
function coerceToString(x) { return x == null ? '' : String(x); }
function safeMatch(s, re) { const t = coerceToString(s); return t ? t.match(re) : null; }
function maskSecret(s, head = 4, tail = 4) {
  const t = coerceToString(s);
  if (t.length <= head + tail + 2) return '****';
  return t.slice(0, head) + '****' + t.slice(-tail);
}

class Client {
  constructor(input = {}) {
    // Normalize cookie input
    const rawCookies = Array.isArray(input.cookie) ? input.cookie : (input.cookie ? [input.cookie] : []);
    const cookieStrings = rawCookies.map(coerceToString).filter(s => s.length > 0);
    if (cookieStrings.length === 0) {
      console.error('Missing required input.cookie array. Provide li_at cookie(s) as an array.');
      throw new Error('Missing required input.cookie array.');
    }
    
    const liAtCookies = cookieStrings.filter(s => safeMatch(s, /li_at=/i));
    if (liAtCookies.length === 0) {
      console.error('No li_at cookie found. Provided cookies (masked):', cookieStrings.map(c => maskSecret(c)));
      throw new Error('No li_at cookie detected in input.cookie. Ensure you supplied li_at.');
    }
    
    // Use the first valid cookie that contains li_at
    this.cookie = liAtCookies[0];
    console.log(`Using LinkedIn cookie (masked): ${maskSecret(this.cookie)}`);

    // ... existing code to process headers, tokens, etc...
  }

  // ... existing methods ...
}

// ... rest of the file ...
