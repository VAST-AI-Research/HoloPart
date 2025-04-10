export class VideoViewerModule {
  constructor(container, videoBaseNames, videoPath, imagePath) {
    this.container = container;
    this.videoBaseNames = videoBaseNames;
    this.videoPath = videoPath;
    this.imagePath = imagePath;
    this.imageExtension = ".png";
    this.videoExtension = ".mp4";
    this.video = null;
  }

  init() {
    this.setupVideo();
    this.createImageSlider();

    document.addEventListener("DOMContentLoaded", () => {
      const videoContainer = document.getElementById("video-container");
      const videoElement = videoContainer.querySelector("video");

      videoElement.addEventListener("loadeddata", () => {
        videoElement.classList.add("loaded");
      });
    });

    this.loadVideo(this.videoBaseNames[0]);
  }

  setupVideo() {
    const viewerContainer = document.querySelector(
      `${this.container} #video-container`
    );

    this.video = document.createElement("video");
    this.video.autoplay = true;
    this.video.controls = true;
    this.video.muted = true;
    this.video.loop = true;
    this.video.playsInline = true;

    viewerContainer.appendChild(this.video);
  }

  loadVideo(baseName) {
    this.video.classList.remove("loaded");
    this.video.src = `${this.videoPath}/${baseName}${this.videoExtension}`;
  }

  createImageSlider() {
    const sliderContainer = document.querySelector(
      `${this.container} #image-slider`
    );
    this.videoBaseNames.forEach((baseName, index) => {
      const slide = document.createElement("div");
      slide.classList.add("swiper-slide");

      const img = document.createElement("img");
      img.src = `${this.imagePath}/${baseName}${this.imageExtension}`;
      img.alt = `Video ${index + 1}`;
      img.onclick = () => this.loadVideo(baseName);

      slide.appendChild(img);
      sliderContainer.appendChild(slide);
    });

    new Swiper(`${this.container} .swiper`, {
      slidesPerView: "auto",
      slidesPerGroup: 2,
      spaceBetween: 10,
      rewind: true,
      navigation: {
        nextEl: `${this.container} .swiper-button-next`,
        prevEl: `${this.container} .swiper-button-prev`,
      },
    });
  }
}
