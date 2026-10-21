// Open a collapsed "开发细节" block when a link or the address points into it.
(function () {
  function reveal() {
    var id = decodeURIComponent(location.hash.slice(1));
    var el = id && document.getElementById(id);
    if (!el) return;
    for (var node = el; node; node = node.parentElement) {
      if (node.tagName === "DETAILS") node.open = true;
    }
    el.scrollIntoView();
  }
  window.addEventListener("hashchange", reveal);
  document.addEventListener("DOMContentLoaded", reveal);
})();
