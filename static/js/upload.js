// Upload UX: show selected filename(s) in the dropzone, and a "processing…"
// state on the submit button when the form is submitted.

(function () {
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]);
    });
  }

  function humanSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  function wireDropzone(label) {
    var input = label.querySelector('input[type=file]');
    var textEl = label.querySelector(".dropzone-text");
    if (!input || !textEl) return;
    var originalHTML = textEl.innerHTML;

    function render() {
      var files = input.files;
      if (!files || files.length === 0) {
        textEl.innerHTML = originalHTML;
        label.classList.remove("selected");
        return;
      }
      if (files.length === 1) {
        var f = files[0];
        textEl.innerHTML =
          '<strong>✓ ' + escapeHtml(f.name) + "</strong>" +
          '<br><small>' + humanSize(f.size) + " · click to change</small>";
      } else {
        var names = [];
        for (var i = 0; i < Math.min(files.length, 3); i++) names.push(files[i].name);
        var more = files.length > 3 ? " …and " + (files.length - 3) + " more" : "";
        textEl.innerHTML =
          "<strong>✓ " + files.length + " files selected</strong>" +
          "<br><small>" + escapeHtml(names.join(", ")) + more + "</small>";
      }
      label.classList.add("selected");
    }

    input.addEventListener("change", render);
  }

  function wireSubmit(form) {
    form.addEventListener("submit", function () {
      var btn = form.querySelector('button[type=submit]');
      if (!btn) return;
      btn.disabled = true;
      btn.dataset.originalText = btn.textContent;
      btn.textContent = "Processing…";
      form.classList.add("submitting");
    });
  }

  document.querySelectorAll("label.dropzone").forEach(wireDropzone);
  document.querySelectorAll("form").forEach(wireSubmit);
})();
