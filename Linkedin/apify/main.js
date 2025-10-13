const Apify = require('apify');
const { Client } = require('./linkedin');

// Input validation and normalization helpers
function buildCookieArray(envOrInput) {
  const raw = envOrInput?.cookie ?? envOrInput?.LI_AT_COOKIE ?? null;
  const arr = Array.isArray(raw) ? raw : (raw ? [raw] : []);
  return arr.map(x => String(x));
}

function buildProxyObject(inp) {
  const proxy = inp?.proxy ?? {};
  if (proxy.useApifyProxy === true) {
    return { 
      useApifyProxy: true, 
      apifyProxyGroups: proxy.apifyProxyGroups || ['RESIDENTIAL'], 
      apifyProxyCountry: proxy.apifyProxyCountry || 'US' 
    };
  }
  if (Array.isArray(proxy.proxyUrls) && proxy.proxyUrls.length > 0) {
    return { useApifyProxy: false, proxyUrls: proxy.proxyUrls };
  }
  // try env-style fallback
  if (process.env.PROXY_URLS) {
    return { 
      useApifyProxy: false, 
      proxyUrls: process.env.PROXY_URLS.split(',').map(s => s.trim()).filter(Boolean) 
    };
  }
  return null;
}

function maskSecret(s, head = 4, tail = 4) {
  const t = s == null ? '' : String(s);
  if (t.length <= head + tail + 2) return '****';
  return t.slice(0, head) + '****' + t.slice(-tail);
}

async function startActorWithRetry(client, actorId, inputPayload, attempts = 3) {
  let lastErr;
  for (let i = 1; i <= attempts; i++) {
    try {
      return await client.actor(actorId).call({ runInput: inputPayload });
    } catch (err) {
      lastErr = err;
      console.warn(`Actor start attempt ${i} failed: ${err.message}`);
      if (i < attempts) await new Promise(r => setTimeout(r, Math.pow(2, i) * 1000));
    }
  }
  throw lastErr;
}

Apify.main(async () => {
  const input = await Apify.getInput();
  
  // Build and validate normalized INPUT_PAYLOAD
  const INPUT_PAYLOAD = {
    cookie: buildCookieArray(input),
    proxy: buildProxyObject(input),
    sourceUrls: Array.isArray(input.sourceUrls) ? input.sourceUrls : 
               (input.sourceUrls ? [input.sourceUrls] : (input.urls || [])),
    deepScrape: input.deepScrape !== false,
    maxDelay: Number(input.maxDelay || 8),
    minDelay: Number(input.minDelay || 2),
    rawData: Boolean(input.rawData),
    limitPerSource: Number(input.limitPerSource || process.env.LIMIT_PER_SOURCE || 50)
  };

  // validate cookie and proxy
  if (!Array.isArray(INPUT_PAYLOAD.cookie) || INPUT_PAYLOAD.cookie.length === 0) {
    console.error('Actor input validation failed: input.cookie is required. Provide li_at in an array (env LI_AT_COOKIE or input.cookie).');
    throw new Error('Missing input.cookie');
  }
  if (!INPUT_PAYLOAD.proxy) {
    console.error('Actor input validation failed: input.proxy is required. Set useApifyProxy or PROXY_URLS.');
    throw new Error('Missing input.proxy');
  }

  // DRY_RUN support
  const isDry = Boolean(input.dryRun || process.env.DRY_RUN === 'true');
  if (isDry) {
    const maskedPayload = {
      ...INPUT_PAYLOAD,
      cookie: INPUT_PAYLOAD.cookie.map(c => maskSecret(c)),
      proxy: INPUT_PAYLOAD.proxy.useApifyProxy 
        ? { 
            useApifyProxy: true, 
            apifyProxyGroups: INPUT_PAYLOAD.proxy.apifyProxyGroups, 
            apifyProxyCountry: INPUT_PAYLOAD.proxy.apifyProxyCountry 
          } 
        : { 
            useApifyProxy: false, 
            proxyUrls: INPUT_PAYLOAD.proxy.proxyUrls.map(u => maskSecret(u)) 
          }
    };
    console.log('DRY_RUN: sanitized INPUT_PAYLOAD:', JSON.stringify(maskedPayload, null, 2));
    process.exit(0);
  }

  // Initialize the Apify API client
  const client = Apify.newClient();
  
  // Set up the request queue and state
  const requestQueue = await Apify.openRequestQueue();
  const sourceUrls = INPUT_PAYLOAD.sourceUrls || [];
  
  // Create a client for the LinkedIn API
  const linkedinClient = new Client(INPUT_PAYLOAD);

  // Process the source URLs
  // ... existing code ...

  try {
    // Use the retry mechanism for actor runs
    const ACTOR_ID = 'YOUR_ACTOR_ID'; // Replace with actual ID if needed
    const run = await startActorWithRetry(client, ACTOR_ID, INPUT_PAYLOAD, 3);
    console.log(`Actor run started successfully: ${run.id}`);
    
    // ... existing processing code ...
    
  } catch (error) {
    console.error('Actor run failed:', error);
    throw error;
  }
});
