// IIFE for top level await
(async () => { 
const visbundle = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js");
const { PoseLandmarker, FilesetResolver, DrawingUtils } = visbundle;

const video = document.getElementById('videoel');
const overlay = document.getElementById('overlay');
const canvas = overlay.getContext('2d');

let poseLandmarker;

let drawImage = true;
let drawLandmarks = true;
let drawConnectors = true;
let zColor = "#FF0066";

function outputMax(mess) {
  window.max.outlet(mess);
}

function outputMaxDict(dstr) {
  window.max.outlet("dictionary", dstr);
}

function setMaxDict(d) {
  window.max.setDict('posedict', d);
}

window.max.bindInlet('draw_image', function (enable) {
  drawImage = enable;
});

window.max.bindInlet('draw_landmarks', function (enable) {
  drawLandmarks = enable;
});

window.max.bindInlet('draw_connectors', function (enable) {
  drawConnectors = enable;
});

window.max.bindInlet('set_mediadevice', async function (deviceLabel) {
  let devices = await getMediaDeviceByLabel(deviceLabel);
  if (!devices.length) {
    window.max.outlet("error", `No video input device: "${deviceLabel}" exists.`);
    return
  }
  const device = devices.shift();
  video.srcObject = await navigator.mediaDevices.getUserMedia({video: {deviceId: device.deviceId}});
});

window.max.bindInlet('get_mediadevices', function () {
  getMediaDevices()
    .then((devices) => {
      let mediadevices = [];
      devices.forEach((device) => {
        if (device.kind === "videoinput") {
          mediadevices.push(device.label);
        }
      });
      window.max.outlet.apply(window.max, ["mediadevices"].concat(mediadevices));
    })
    .catch((err) => {
      window.max.outlet("error",`${err.name}: ${err.message}`);
    });
});

const getMediaDevices = async () => {   
  if (!navigator.mediaDevices?.enumerateDevices) {
    window.max.outlet("error", "Cannot list available media devices.");
    return []
  }
  return await navigator.mediaDevices.enumerateDevices();
}

const getMediaDeviceByLabel = async (deviceLabel) => {
  let mediaDevices = await getMediaDevices();
  return mediaDevices.filter(device => device.label == deviceLabel);
}

let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvas);

function onResultsPose(results) {

  canvas.save();
  canvas.clearRect(0, 0, overlay.width, overlay.height);

  if(drawImage) {
    canvas.drawImage(
      results.image, 0, 0, overlay.width, overlay.height);
  }

  const output = {
    "left":{},
    "right":{},
    "neutral":{}
  };
  Object.values(POSE_LANDMARKS_LEFT).forEach(([landmark, index]) => { output["left"][landmark] = results.landmarks[0][index] });
  Object.values(POSE_LANDMARKS_RIGHT).forEach(([landmark, index]) => { output["right"][landmark] = results.landmarks[0][index] });
  Object.values(POSE_LANDMARKS_NEUTRAL).forEach(([landmark, index]) => { output["neutral"][landmark] = results.landmarks[0][index] });
  setMaxDict(output);

  for (const landmark of results.landmarks) {
    if (drawLandmarks) {
      drawingUtils.drawLandmarks(landmark, {
        radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 1, 0.5),
        color: zColor, fillColor: '#FF0000'
      });
    }
    if (drawConnectors) {
      drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {
        color: (data) => {
          const x0 = overlay.width * data.from.x;
          const y0 = overlay.height * data.from.y;
          const x1 = overlay.width * data.to.x;
          const y1 = overlay.height * data.to.y;

          const z0 = clamp(data.from.z + 0.5, 0, 1);
          const z1 = clamp(data.to.z + 0.5, 0, 1);

          const gradient = canvas.createLinearGradient(x0, y0, x1, y1);
          gradient.addColorStop(
              0, `rgba(0, ${255 * z0}, ${255 * (1 - z0)}, 1)`);
          gradient.addColorStop(
              1.0, `rgba(0, ${255 * z1}, ${255 * (1 - z1)}, 1)`);
          return gradient;
        }}
      );
    }
  }

  outputMax("update");
  canvas.restore();
}

const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
);
poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
    delegate: "GPU"
  },
  numPoses: 1,
  runningMode: "VIDEO"
})


const camera = new Camera(video, {
  onFrame: async () => {
    let nowInMs = Date.now();
    if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;
      results = poseLandmarker.detectForVideo(video, nowInMs);
      results.image = video;
      onResultsPose(results);
    }
  },
  width: 640,
  height: 480
});
camera.start();

})(); // end IIF