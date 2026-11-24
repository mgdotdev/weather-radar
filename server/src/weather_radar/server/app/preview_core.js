/* Shared engine for the Home Assistant preview pages (/ha-preview/, /graph-preview/).

   A page supplies what is specific to its subject (how a URL maps to state and
   back, its controls); this file does everything else:

     * the iframe, sized as a Home Assistant card / panel / device;
     * the URL + card-YAML outputs (base URL, optional token, copy buttons);
     * applying edits to the iframe (debounced) and "follow" — reading the
       preview's own URL back as you change things inside it;
     * "paste a URL", the saved-in-the-address-bar state, and a test hook.

   cfg = {
     hook:          name of the window property exposed for tests / debugging,
     pathRe:        RegExp matching the pathname of the page being previewed,
     controlIds:    ids of inputs whose input/change events mean "state edited",
     defaultState():                  -> a fresh state object
     fromUrl(raw):                    -> state | null   (absolute or relative URL)
     toRelative(state, token):        -> "/path?query"
     renderControls(state, skipFocused),
     readControls(state)              -- copy the controls' values INTO state
     wire(api)                        -- optional: page-specific buttons
     live:          optional { keys: [...], apply(win, state, changedKeys) -> bool }
                    URL parameters that are purely VISUAL and can be changed inside
                    the loaded page without reloading it (it exposes window.WRLive);
                    apply() performs the change and returns false if it can't.
   }
   api = { state (getter), set(S), render(skipFocused), changed(immediate) } */
(function () {
  const $ = (id) => document.getElementById(id)

  function start(cfg) {
    const frame = $("frame")
    const pageToken = new URL(window.location.href).searchParams.get("token") || ""
    let P = cfg.defaultState()

    // lastApplied: the URL the frame shows (or is about to). reloading: a src
    // change is in flight. pendingEdit: a control was edited and its (debounced)
    // reload hasn't been issued yet — until then the controls, not the frame,
    // are the truth. suppressRead: a sync is rewriting the controls.
    let lastApplied = "", reloading = false, reloadTimer = null
    let pendingEdit = false, suppressRead = false

    // "Sample data": the frame asks the API for canned data (fixture=1) instead of
    // running the real pipeline — instant and offline. The flag is added to the
    // FRAME's url only; everything this page shows or copies (URL, YAML, hash,
    // comparisons) is without it.
    const withFixture = (rel) => {
      if (!$("fixture").checked) return rel
      const u = new URL(rel, window.location.origin); u.searchParams.set("fixture", "1")
      return u.pathname + u.search
    }
    const withoutFixture = (rel) => {
      const u = new URL(rel, window.location.origin); u.searchParams.delete("fixture")
      return u.pathname + u.search
    }
    // A URL in canonical form: parsed and rebuilt by this page, so two URLs that
    // MEAN the same thing compare equal — the previewed page rewrites its own URL
    // (e.g. the map adds `speed=1.00`, which the controls omit as the default; a
    // number can be written 2 or 2.0).
    function canonical(rel) {
      const S = cfg.fromUrl(rel)
      return S ? cfg.toRelative(S, "") : rel
    }
    // Which parameters differ between two relative URLs, by MEANING ("#path" if
    // the page does).
    function changedParams(a, b) {
      const ua = new URL(canonical(a), window.location.origin), ub = new URL(canonical(b), window.location.origin)
      if (ua.pathname !== ub.pathname) return ["#path"]
      const keys = new Set([...ua.searchParams.keys(), ...ub.searchParams.keys()])
      keys.delete("token"); keys.delete("fixture")
      return [...keys].filter((k) => ua.searchParams.get(k) !== ub.searchParams.get(k))
    }

    const api = {
      get state() { return P },
      set(S) { P = S },
      render(skipFocused) { cfg.renderControls(P, !!skipFocused) },
      changed(immediate) { changed(immediate) },
    }

    // ---- applying changes to the iframe --------------------------------
    // The fast path. When ONLY visual settings changed (pin size, legend size, lock,
    // …) and the loaded page can apply them in place, do that — no reload, no data
    // fetch, instant. Anything else (area, layer, point, curves…), or any failure,
    // returns false and the caller reloads as before.
    function tryLive() {
      if (!cfg.live || reloading) return false
      let win
      try { win = frame.contentWindow; if (!win || !win.WRLive) return false } catch (_) { return false }
      const rel = cfg.toRelative(P, pageToken)
      const changed = changedParams(lastApplied, rel)
      if (!changed.length) { lastApplied = frameRelativeNow() || rel; return true }   // nothing differs
      if (!changed.every((k) => cfg.live.keys.includes(k))) return false
      try { if (!cfg.live.apply(win, P, changed)) return false } catch (_) { return false }
      lastApplied = frameRelativeNow() || rel          // what the frame now shows
      return true
    }

    function applyToFrame(immediate) {
      clearTimeout(reloadTimer)
      if (tryLive()) { pendingEdit = false; return }
      const go = () => {
        pendingEdit = false
        const rel = cfg.toRelative(P, pageToken)       // the frame's auth is the page's own token, if any
        if (rel === lastApplied) return
        lastApplied = rel; reloading = true
        frame.src = withFixture(rel)
      }
      if (immediate) go(); else reloadTimer = setTimeout(go, 400)
    }

    function changed(immediate) {
      pendingEdit = true
      if (!suppressRead) cfg.readControls(P)
      outputs(); saveHash(); applyToFrame(immediate)
    }

    // ---- outputs ---------------------------------------------------------
    function absUrl() {
      const tok = $("useToken").checked ? ($("token").value.trim() || pageToken) : ""
      const base = ($("base").value.trim() || window.location.origin).replace(/\/+$/, "")
      return base + cfg.toRelative(P, tok)
    }
    function heightFromAspect(w, ar) {
      if (!ar) return Math.round(w * 0.5)                   // Home Assistant's default: half the width
      if (ar.endsWith("%")) return Math.round(w * parseFloat(ar) / 100)
      const [a, b] = ar.split(":").map(Number)
      return Math.round(w * (b || 1) / a)
    }
    function frameSize() {
      const mode = $("mode").value
      if (mode === "card") { const w = +$("fw").value || 480; return [w, heightFromAspect(w, $("ar").value)] }
      if (mode === "device") return $("dev").value.split("x").map(Number)
      return [+$("fw").value || 480, +$("fh").value || 320]
    }
    function outputs() {
      const url = absUrl()
      $("urlOut").textContent = url
      const mode = $("mode").value
      let y = "type: iframe\nurl: " + JSON.stringify(url)
      if (mode === "card") y += $("ar").value ? `\naspect_ratio: ${$("ar").value}` : ""
      else if (mode === "sections") y += "\n# sections view: the card's size comes from the grid (aspect_ratio is ignored)"
      else if (mode === "panel") y += "\n# use this as the single card of a Panel (1 card) view"
      $("yamlOut").textContent = y
    }

    // ---- frame sizing ----------------------------------------------------
    function layoutFrame() {
      const mode = $("mode").value
      $("arLbl").style.display = mode === "card" ? "" : "none"
      $("hLbl").style.display = (mode === "sections" || mode === "panel") ? "" : "none"
      $("wLbl").style.display = mode === "device" ? "none" : ""
      $("devLbl").style.display = mode === "device" ? "" : "none"
      const [w, h] = frameSize()
      frame.style.width = w + "px"; frame.style.height = h + "px"
      const wrap = $("frameWrap"), stage = $("stage")
      let scale = 1
      if ($("fit").checked) {
        scale = Math.max(0.2, Math.min(1, (stage.clientWidth - 34) / w, (stage.clientHeight - 60) / h))
      }
      wrap.style.transform = scale === 1 ? "" : `scale(${scale})`
      wrap.style.marginBottom = scale === 1 ? "" : `${-(h * (1 - scale))}px`
      wrap.style.marginRight = scale === 1 ? "" : `${-(w * (1 - scale))}px`
      $("info").textContent = `${w} × ${h} px` + (scale < 1 ? `  (shown at ${Math.round(scale * 100)}% to fit)` : "")
    }

    // ---- follow the preview ------------------------------------------------
    // The preview is same-origin, so its address bar (which the page keeps
    // current as you drag/edit things inside it) can be read back. While
    // "follow" is on, those changes flow into the controls — set it up by hand
    // inside the frame, then copy the URL — WITHOUT reloading the frame.
    function frameRelativeNow() {
      try {
        const l = frame.contentWindow.location
        return cfg.pathRe.test(l.pathname) ? withoutFixture(l.pathname + l.search) : null
      } catch (_) { return null }
    }
    function stripToken(rel) {
      const u = new URL(rel, window.location.origin); u.searchParams.delete("token")
      return u.pathname + u.search
    }
    function pollFrame() {
      // Never pull from the frame while an edit is waiting to be sent to it (or a
      // reload is in flight): the frame still shows the OLD url, and adopting it
      // would silently throw the edit away.
      if (!$("follow").checked || reloading || pendingEdit) return
      const cur = frameRelativeNow()
      if (!cur || cur === lastApplied) return
      const S = cfg.fromUrl(cur)
      if (!S) return
      lastApplied = cur                       // the frame already shows this: no reload
      P = S
      suppressRead = true
      cfg.renderControls(P, true); outputs(); saveHash()
      suppressRead = false
      $("status").textContent = "✓ synced"
      setTimeout(() => { $("status").textContent = "" }, 1200)
    }
    setInterval(pollFrame, 700)
    frame.addEventListener("load", () => {
      reloading = false
      const cur = frameRelativeNow()
      if (cur) lastApplied = cur
    })

    // ---- persistence of THIS page's settings (hash) -------------------------
    function saveHash() {
      const o = { u: stripToken(cfg.toRelative(P, "")), mode: $("mode").value, fw: $("fw").value, fh: $("fh").value,
                  ar: $("ar").value, dev: $("dev").value, fit: $("fit").checked, follow: $("follow").checked,
                  base: $("base").value, fx: $("fixture").checked }
      history.replaceState(null, "", window.location.pathname + window.location.search + "#" + encodeURIComponent(JSON.stringify(o)))
    }

    // ---- wiring ------------------------------------------------------------
    for (const id of cfg.controlIds) {
      $(id).addEventListener("input", () => changed(false))
      $(id).addEventListener("change", () => changed(true))
    }
    function importFrom(raw) {
      const S = cfg.fromUrl(raw.trim())
      if (!S) { $("status").textContent = "not a recognised URL"; return }
      P = S; cfg.renderControls(P, false); changed(true)
    }
    $("importBtn").addEventListener("click", () => importFrom($("importUrl").value))
    $("importUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") importFrom($("importUrl").value) })
    for (const id of ["mode", "fw", "fh", "ar", "dev", "fit"]) {
      $(id).addEventListener("input", () => { layoutFrame(); outputs(); saveHash() })
    }
    // "common widths": a phone's HA card is ~16 px narrower than the screen, so the map is cropped more.
    $("wPreset").addEventListener("change", () => {
      const w = $("wPreset").value
      if (w) { $("fw").value = w; $("fw").dispatchEvent(new Event("input", { bubbles: true })) }
      $("wPreset").value = ""
    })
    for (const id of ["base", "useToken", "token"]) $(id).addEventListener("input", () => { outputs(); saveHash() })
    $("follow").addEventListener("input", saveHash)
    $("fixture").addEventListener("input", () => { saveHash(); lastApplied = ""; applyToFrame(true) })
    $("reload").addEventListener("click", () => { lastApplied = ""; applyToFrame(true) })
    const copy = async (text, btn) => {
      try { await navigator.clipboard.writeText(text) } catch (_) {
        const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta)
        ta.select(); document.execCommand("copy"); ta.remove()
      }
      const old = btn.textContent; btn.textContent = "Copied ✓"; setTimeout(() => { btn.textContent = old }, 1200)
    }
    $("copyUrl").addEventListener("click", (e) => copy($("urlOut").textContent, e.target))
    $("copyYaml").addEventListener("click", (e) => copy($("yamlOut").textContent, e.target))
    $("openTab").addEventListener("click", () => window.open(absUrl(), "_blank"))
    window.addEventListener("resize", layoutFrame)
    // the links between the preview pages keep the access token
    if (pageToken) for (const a of document.querySelectorAll("a.nav")) {
      const u = new URL(a.getAttribute("href"), window.location.origin); u.searchParams.set("token", pageToken)
      a.setAttribute("href", u.pathname + u.search)
    }
    if (cfg.wire) cfg.wire(api)

    // ---- startup -------------------------------------------------------------
    $("fw").value = 480; $("fh").value = 320; $("fit").checked = true; $("follow").checked = true
    $("fixture").checked = true
    $("base").value = window.location.origin
    $("token").value = pageToken; $("useToken").checked = !!pageToken
    try {
      const o = JSON.parse(decodeURIComponent(window.location.hash.slice(1)))
      for (const k of ["mode", "fw", "fh", "ar", "dev", "base"]) if (o[k] != null) $(k).value = o[k]
      if (o.fit != null) $("fit").checked = o.fit
      if (o.follow != null) $("follow").checked = o.follow
      if (o.fx != null) $("fixture").checked = o.fx
      if (o.u) { const S = cfg.fromUrl(o.u); if (S) P = S }
    } catch (_) {}
    const qUrl = new URL(window.location.href).searchParams.get("u")   // ?u=<url> also works
    if (qUrl) { const S = cfg.fromUrl(qUrl); if (S) P = S }
    cfg.renderControls(P, false); layoutFrame(); outputs()
    applyToFrame(true)
    window[cfg.hook] = { get state() { return P }, fromUrl: cfg.fromUrl, toRelative: cfg.toRelative, pollFrame }
  }

  window.WRPreview = { start }
})()
