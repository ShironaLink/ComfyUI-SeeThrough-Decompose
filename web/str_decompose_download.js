import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI-SeeThrough-Decompose.SavePSD",
    async nodeCreated(node) {
        if (node.comfyClass !== "STR_SaveDecomposedPSD") return;

        // "Download PSD" button
        const dlBtn = node.addWidget("button", "Download PSD", "Download PSD", async () => {
            try {
                // Read log file to get latest PSD info
                const t = Date.now();
                const logResp = await fetch(
                    api.apiURL(`/view?filename=str_decompose_info.log&type=output&t=${t}`)
                );
                if (!logResp.ok) {
                    alert("PSD info not found. Run the workflow first.");
                    return;
                }
                const infoFilename = (await logResp.text()).trim();

                // Fetch layer info JSON
                const infoResp = await fetch(
                    api.apiURL(`/view?filename=${infoFilename}&type=output&t=${t}`)
                );
                if (!infoResp.ok) {
                    alert("Layer info JSON not found: " + infoFilename);
                    return;
                }
                const layerInfo = await infoResp.json();

                // Load ag-psd library
                await loadAgPsd();

                // Build PSD
                const psd = await buildPSD(layerInfo);

                // Write and download
                const buffer = window.AgPsd.writePsd(psd);
                const blob = new Blob([buffer], { type: "application/octet-stream" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = (layerInfo.prefix || "decomposed") + ".psd";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } catch (e) {
                console.error("[STR_Decompose] Download error:", e);
                alert("Download failed: " + e.message);
            }
        });
        dlBtn.serialize = false;

        // Style the button
        const origDraw = dlBtn.draw;
        dlBtn.draw = function(ctx, node, width, y, height) {
            ctx.fillStyle = "#10B981";
            ctx.strokeStyle = "#059669";
            if (origDraw) origDraw.call(this, ctx, node, width, y, height);
        };
    },
});

// Load ag-psd bundle (reuse See-through's if available)
let agPsdLoaded = false;
async function loadAgPsd() {
    if (agPsdLoaded || window.AgPsd) {
        agPsdLoaded = true;
        return;
    }

    // Try See-through's bundle first
    const paths = [
        "/extensions/ComfyUI-See-through/ag-psd.bundle.js",
        "/extensions/ComfyUI-SeeThrough-Decompose/ag-psd.bundle.js",
    ];

    for (const path of paths) {
        try {
            const resp = await fetch(path, { method: "HEAD" });
            if (resp.ok) {
                await new Promise((resolve, reject) => {
                    const script = document.createElement("script");
                    script.src = path;
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                });
                agPsdLoaded = true;
                return;
            }
        } catch (e) {
            continue;
        }
    }

    throw new Error("ag-psd library not found. Make sure ComfyUI-See-through is installed.");
}

// Build PSD from layer info
async function buildPSD(layerInfo) {
    const { width, height, layers } = layerInfo;

    const psd = {
        width: width,
        height: height,
        children: [],
    };

    // Create composite canvas
    const compositeCanvas = document.createElement("canvas");
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext("2d");

    // Blend mode mapping
    const blendModes = {
        "flat": "normal",
        "shadow": "multiply",
        "highlight": "screen",
        "lineart": "normal",
    };

    // Load each layer
    for (const layer of layers) {
        const imgResp = await fetch(
            api.apiURL(`/view?filename=${layer.filename}&type=output&t=${Date.now()}`)
        );
        const imgBlob = await imgResp.blob();
        const imgUrl = URL.createObjectURL(imgBlob);

        const img = new Image();
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = imgUrl;
        });

        // Draw to canvas
        const layerCanvas = document.createElement("canvas");
        layerCanvas.width = width;
        layerCanvas.height = height;
        const layerCtx = layerCanvas.getContext("2d");
        layerCtx.drawImage(img, layer.left || 0, layer.top || 0);

        // Add to composite
        compositeCtx.drawImage(layerCanvas, 0, 0);

        // Determine blend mode from layer type suffix
        let blendMode = "normal";
        for (const [suffix, mode] of Object.entries(blendModes)) {
            if (layer.name.endsWith("_" + suffix)) {
                blendMode = mode;
                break;
            }
        }

        psd.children.push({
            name: layer.name,
            canvas: layerCanvas,
            blendMode: blendMode,
            opacity: 1,
            left: layer.left || 0,
            top: layer.top || 0,
        });

        URL.revokeObjectURL(imgUrl);
    }

    psd.canvas = compositeCanvas;
    return psd;
}
