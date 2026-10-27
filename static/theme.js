/* ============================================================================
   Light/dark switch shared by the portal, the demo and the API docs.
   Loaded in <head> before the stylesheet so the theme is applied before the
   first paint (no flash of the wrong palette).

   Order of precedence: saved choice -> OS preference -> dark (brand default).
   Any element with id="themeBtn" becomes the toggle; theme.css styles it.
   ========================================================================== */
(function () {
  var KEY = 'ocr-theme';

  function saved() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }

  function prefersLight() {
    return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches);
  }

  function label(theme) {
    var btn = document.getElementById('themeBtn');
    if (!btn) return;
    var next = theme === 'light' ? 'Dark mode' : 'Light mode';
    btn.textContent = next;
    btn.setAttribute('aria-label', 'Switch to ' + next.toLowerCase());
    btn.setAttribute('title', 'Switch to ' + next.toLowerCase());
  }

  function apply(theme) {
    document.documentElement.dataset.theme = theme;
    label(theme);
    // Same-origin embeds (the portal's "Sample use case" frame) follow along.
    // An embedded page that loads later reads the saved choice itself.
    var frames = document.querySelectorAll('iframe');
    for (var i = 0; i < frames.length; i++) {
      try {
        var doc = frames[i].contentDocument;
        if (doc && doc.documentElement) doc.documentElement.dataset.theme = theme;
      } catch (e) { /* cross-origin frame: leave it alone */ }
    }
  }

  window.toggleTheme = function () {
    var next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    try { localStorage.setItem(KEY, next); } catch (e) { /* private mode: session only */ }
    apply(next);
  };

  // Runs during <head> parsing: sets the palette even before <body> exists.
  apply(saved() || (prefersLight() ? 'light' : 'dark'));

  function bind() {
    // Re-label here because the pill did not exist when apply() first ran.
    label(document.documentElement.dataset.theme || 'dark');
    // Delegated, so the pill keeps working if the nav is ever re-rendered.
    document.addEventListener('click', function (e) {
      for (var el = e.target; el && el !== document; el = el.parentElement) {
        if (el.id === 'themeBtn') {
          e.preventDefault();
          window.toggleTheme();
          return;
        }
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
