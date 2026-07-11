document.documentElement.classList.add("js");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function safePlay(video, force = false) {
  if (!video || (!force && reducedMotion) || video.closest("[hidden]")) return;

  const promise = video.play();
  if (promise && typeof promise.catch === "function") {
    promise.catch(() => updateVideoButton(video, false));
  }
}

function updateVideoButton(video, isPlaying) {
  const card = video.closest(".video-card");
  const button = card?.querySelector(".video-toggle");
  if (!button) return;

  const title = card.querySelector("h3")?.textContent?.trim() || "rollout";
  button.querySelector("span").textContent = isPlaying ? "Ⅱ" : "▶";
  button.setAttribute("aria-label", `${isPlaying ? "Pause" : "Play"} ${title} video`);
}

function initHeader() {
  const header = document.querySelector("[data-header]");
  const toggle = document.querySelector("[data-nav-toggle]");
  const links = document.querySelector("[data-nav-links]");
  if (!header || !toggle || !links) return;

  const updateHeader = () => header.classList.toggle("is-scrolled", window.scrollY > 18);
  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  const closeMenu = () => {
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", "Open navigation");
    links.classList.remove("is-open");
    header.classList.remove("is-open");
    document.body.classList.remove("nav-open");
  };

  toggle.addEventListener("click", () => {
    const opening = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(opening));
    toggle.setAttribute("aria-label", opening ? "Close navigation" : "Open navigation");
    links.classList.toggle("is-open", opening);
    header.classList.toggle("is-open", opening);
    document.body.classList.toggle("nav-open", opening);
  });

  links.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
  window.addEventListener("resize", () => {
    if (window.innerWidth > 860) closeMenu();
  });
}

function initReveal() {
  const items = [...document.querySelectorAll("[data-reveal]")];
  items.forEach((item) => {
    item.style.setProperty("--reveal-delay", `${item.dataset.delay || 0}ms`);
  });

  if (reducedMotion || !("IntersectionObserver" in window)) {
    items.forEach((item) => item.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
  );

  items.forEach((item) => observer.observe(item));
}

function initVideos() {
  const videos = [...document.querySelectorAll("[data-autoplay], [data-gallery-video]")];

  videos.forEach((video) => {
    video.muted = true;
    video.addEventListener("play", () => updateVideoButton(video, true));
    video.addEventListener("pause", () => updateVideoButton(video, false));
  });

  if (!reducedMotion && "IntersectionObserver" in window) {
    const videoObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const video = entry.target;
          video.dataset.inView = entry.isIntersecting ? "true" : "false";
          if (entry.isIntersecting && !video.closest("[hidden]")) {
            safePlay(video);
          } else {
            video.pause();
          }
        });
      },
      { threshold: 0.28 },
    );

    videos.forEach((video) => videoObserver.observe(video));
  }

  document.querySelectorAll(".video-card").forEach((card) => {
    const video = card.querySelector("video");
    const button = card.querySelector(".video-toggle");
    if (!video || !button) return;

    const togglePlayback = () => {
      if (video.paused) safePlay(video, true);
      else video.pause();
    };

    button.addEventListener("click", togglePlayback);
    video.addEventListener("click", togglePlayback);
    updateVideoButton(video, false);
  });

  window.addEventListener("pagehide", () => videos.forEach((video) => video.pause()));
}

function initRolloutFilters() {
  const tabs = [...document.querySelectorAll("[data-filter]")];
  const cards = [...document.querySelectorAll(".video-card[data-category]")];
  const count = document.querySelector("[data-visible-count]");
  if (!tabs.length || !cards.length) return;

  const applyFilter = (filter) => {
    let visible = 0;

    cards.forEach((card) => {
      const shouldShow =
        filter === "featured" ? card.dataset.featured === "true" : card.dataset.category === filter;
      const video = card.querySelector("video");
      card.hidden = !shouldShow;

      if (shouldShow) {
        visible += 1;
        card.classList.remove("filter-enter");
        window.requestAnimationFrame(() => card.classList.add("filter-enter"));
        if (video?.dataset.inView === "true") safePlay(video);
      } else if (video) {
        video.pause();
      }
    });

    if (count) count.textContent = String(visible);
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => {
      tabs.forEach((item) => {
        const active = item === tab;
        item.setAttribute("aria-selected", String(active));
        item.tabIndex = active ? 0 : -1;
      });
      applyFilter(tab.dataset.filter);
    });

    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const next = (index + direction + tabs.length) % tabs.length;
      tabs[next].focus();
      tabs[next].click();
    });
  });

  tabs.forEach((tab, index) => (tab.tabIndex = index === 0 ? 0 : -1));
}

function initLightbox() {
  const dialog = document.querySelector("[data-lightbox-dialog]");
  const image = dialog?.querySelector("[data-lightbox-image]");
  const caption = dialog?.querySelector("[data-lightbox-caption]");
  const close = dialog?.querySelector("[data-lightbox-close]");
  if (!dialog || !image || !caption || !close) return;

  document.querySelectorAll("[data-lightbox]").forEach((button) => {
    button.addEventListener("click", () => {
      image.src = button.dataset.lightbox;
      image.alt = button.querySelector("img")?.alt || "Expanded research figure";
      caption.textContent = button.dataset.caption || "";
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
      document.body.classList.add("nav-open");
      close.focus();
    });
  });

  const closeDialog = () => {
    if (typeof dialog.close === "function") dialog.close();
    else dialog.removeAttribute("open");
    document.body.classList.remove("nav-open");
  };

  close.addEventListener("click", closeDialog);
  dialog.addEventListener("close", () => document.body.classList.remove("nav-open"));
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) closeDialog();
  });
}

function initCopyCitation() {
  const button = document.querySelector("[data-copy-bib]");
  const citation = document.querySelector("#bibtex code");
  if (!button || !citation) return;

  const writeFallback = (text) => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  };

  button.addEventListener("click", async () => {
    const text = citation.textContent.trim();
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else writeFallback(text);
      button.textContent = "Copied";
    } catch {
      writeFallback(text);
      button.textContent = "Copied";
    }
    window.setTimeout(() => (button.textContent = "Copy citation"), 1800);
  });
}

initHeader();
initReveal();
initVideos();
initRolloutFilters();
initLightbox();
initCopyCitation();
