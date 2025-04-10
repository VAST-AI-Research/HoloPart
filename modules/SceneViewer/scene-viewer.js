export class ViewerModule {
  constructor(container, modelBaseNames, modelPath, imagePath) {
    this.container = container;
    this.modelBaseNames = modelBaseNames;
    this.modelPath = modelPath;
    this.imagePath = imagePath;
    this.imageExtension = ".png";
    this.modelExtension = ".glb";
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.model = null;
    this.controls = null;
  }

  init() {
    this.setupScene();
    this.createImageSlider();
    this.loadModel(this.modelBaseNames[0]);
  }

  setupScene() {
    const viewerContainer = document.querySelector(
      `${this.container} #viewer-container`
    );
    const width = viewerContainer.clientWidth;
    const height = viewerContainer.clientHeight;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 1000);
    this.camera.position.set(0, 1, 5);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setClearColor(0xffffff);
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.renderer.physicallyCorrectLights = true;
    viewerContainer.appendChild(this.renderer.domElement);

    this.controls = new THREE.OrbitControls(
      this.camera,
      this.renderer.domElement
    );
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.25;

    // Increase directional light intensity
    const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
    directionalLight.position.set(5, 10, 7);
    this.scene.add(directionalLight);

    // Increase point light intensity
    const lightIntensity = 25;
    const lightDistance = 100;

    const directions = [
      [10, 0, 0], // +x
      [-10, 0, 0], // -x
      [0, 10, 0], // +y
      [0, -10, 0], // -y
      [0, 0, 10], // +z
      [0, 0, -10], // -z
    ];

    directions.forEach((dir, index) => {
      const pointLight = new THREE.PointLight(
        0xffffff,
        lightIntensity,
        lightDistance
      );
      pointLight.position.set(...dir);
      pointLight.castShadow = true;
      this.scene.add(pointLight);

      pointLight.name = `PointLight_${index}`;
    });

    window.addEventListener("resize", () => {
      const newWidth = viewerContainer.clientWidth;
      const newHeight = viewerContainer.clientHeight;
      this.renderer.setSize(newWidth, newHeight);
      this.camera.aspect = newWidth / newHeight;
      this.camera.updateProjectionMatrix();
    });

    this.animate();
  }

  loadModel(baseName, index) {
    if (this.model) this.scene.remove(this.model);

    const overlay = document.querySelector(
      `${this.container} #loading-overlay`
    );
    overlay.style.display = "flex";

    const loader = new THREE.GLTFLoader();
    loader.load(
      `${this.modelPath}/${baseName}${this.modelExtension}`,
      (gltf) => {
        this.model = gltf.scene;
        this.scene.add(this.model);

        this.model.traverse((child) => {
          if (child.isMesh) child.visible = true;
        });

        this.changeModelColor(0xffffff);

        // Reset camera
        this.camera.position.set(0, 1, 5);

        // Replace buttons with explode slider
        this.createExplodeSlider();

        overlay.style.display = "none";
      }
    );
  }

  changeModelColor(color) {
    if (this.model) {
      this.model.traverse((child) => {
        if (child.isMesh) {
          child.material.color.set(color);
        }
      });
    }
  }

  createImageSlider() {
    const sliderContainer = document.querySelector(
      `${this.container} #image-slider`
    );
    this.modelBaseNames.forEach((baseName, index) => {
      const slide = document.createElement("div");
      slide.classList.add("swiper-slide");

      const img = document.createElement("img");
      img.src = `${this.imagePath}/${baseName}${this.imageExtension}`;
      img.alt = `Model ${index + 1}`;
      img.onclick = () => this.loadModel(baseName, index);

      slide.appendChild(img);
      sliderContainer.appendChild(slide);
    });

    this.swiper = new Swiper(`${this.container} .swiper`, {
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

  createExplodeSlider() {
    const controlsDiv = document.querySelector(
      `${this.container} #button-block`
    );
    controlsDiv.innerHTML = ""; // Clear existing buttons

    const sliderContainer = document.createElement("div");
    sliderContainer.style.display = "flex";
    sliderContainer.style.alignItems = "center";
    sliderContainer.style.justifyContent = "center";
    sliderContainer.style.margin = "10px";

    const label = document.createElement("span");
    label.textContent = "Explode: ";
    label.style.marginRight = "10px";
    label.style.fontWeight = "bold"; // Make the label bold

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.01";
    slider.value = "0";
    slider.style.width = "300px";

    slider.oninput = (event) => {
      const explodeAmount = parseFloat(event.target.value);
      this.applyExplodeEffect(explodeAmount);
    };

    sliderContainer.appendChild(label);
    sliderContainer.appendChild(slider);
    controlsDiv.appendChild(sliderContainer);
  }

  applyExplodeEffect(explodeAmount) {
    if (!this.model) return;

    const root = this.model.children[0];
    root.children.forEach((part, index) => {
      const bbox = new THREE.Box3().setFromObject(part);
      const center = bbox.getCenter(new THREE.Vector3());
      const direction = center.clone().sub(this.scene.position).normalize();

      // Calculate the new position based on the explode amount
      const originalPosition = new THREE.Vector3().copy(part.userData.originalPosition || part.position);
      const offset = direction.multiplyScalar(explodeAmount * 2);
      const newPosition = originalPosition.clone().add(offset);

      // Store the original position if not already stored
      if (!part.userData.originalPosition) {
        part.userData.originalPosition = originalPosition.clone();
      }

      part.position.copy(newPosition);
    });
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
