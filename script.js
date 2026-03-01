document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('vision-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width, height;
    const resizeCanvas = () => {
        const rect = canvas.parentElement.getBoundingClientRect();
        width = canvas.width = rect.width;
        height = canvas.height = rect.height;
    };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    const bgImage = new Image();
    bgImage.src = 'bg.jpg';

    // Define relative static "detected" objects to highlight in the image [rx, ry, rw, rh] are percentages
    // Added timeOffset to stagger their appearance
    const detectedObjects = [
        { label: 'PERSON', rx: 0.35, ry: 0.45, rw: 0.15, rh: 0.40, confidence: 96, timeOffset: 0 },
        { label: 'CAR', rx: 0.60, ry: 0.55, rw: 0.25, rh: 0.25, confidence: 92, timeOffset: 40 },
        { label: 'PERSON', rx: 0.10, ry: 0.40, rw: 0.12, rh: 0.55, confidence: 88, timeOffset: 80 },
        { label: 'TRAFFIC_LIGHT', rx: 0.23, ry: 0.28, rw: 0.05, rh: 0.15, confidence: 99, timeOffset: 120 },
        { label: 'TRASH_RECEPTACLE', rx: 0.85, ry: 0.65, rw: 0.08, rh: 0.20, confidence: 85, timeOffset: 160 }
    ];

    let time = 0;
    let scanY = 0;

    const terminal = document.getElementById('terminal-readout');
    const messages = [
        "VLM: Semantic extraction complete. Confidence 98%.",
        "YOLOv11seg: 5 dynamic obstacles identified in trajectory.",
        "Haptic Matrix: Actuators firing spatially [Left-Front].",
        "Depth perception map updated. Latency < 16ms.",
        "VLM: Approaching pedestrian crossing.",
        "System: Recalibrating spatial mesh.",
        "YOLOv11seg: Contextual segments isolated."
    ];
    let msgIndex = 0;

    setInterval(() => {
        msgIndex = (msgIndex + 1) % messages.length;
        terminal.innerText = `> ${messages[msgIndex]}`;
        terminal.style.textShadow = '2px 0 0 #fff, -2px 0 0 rgba(255,255,255,0.5)';
        setTimeout(() => {
            terminal.style.textShadow = 'none';
        }, 100);
    }, 3500);

    function render() {
        ctx.clearRect(0, 0, width, height);

        if (bgImage.complete && bgImage.naturalWidth > 0) {
            const aspect = bgImage.naturalWidth / bgImage.naturalHeight;
            const canvasAspect = width / height;

            let drawWidth = width;
            let drawHeight = height;
            let offsetX = 0;
            let offsetY = 0;

            if (canvasAspect > aspect) {
                drawHeight = width / aspect;
                offsetY = (height - drawHeight) / 2;
            } else {
                drawWidth = height * aspect;
                offsetX = (width - drawWidth) / 2;
            }

            // Draw original image
            ctx.drawImage(bgImage, offsetX, offsetY, drawWidth, drawHeight);

            // Dim background significantly to make highlights pop
            ctx.fillStyle = 'rgba(5, 5, 5, 0.7)';
            ctx.fillRect(0, 0, width, height);

            // Jitter for a fake live-feed effect
            const jitterX = Math.sin(time * 0.15) * 1.5;
            const jitterY = Math.cos(time * 0.12) * 1.5;

            detectedObjects.forEach(obj => {
                // Determine if this object should be visible yet based on elapsed frames
                if (time < obj.timeOffset) return;

                // Pop-in animation scale
                let progress = Math.min((time - obj.timeOffset) / 20, 1.0);
                // Simple easing out
                let scale = 1.0 - Math.pow(1.0 - progress, 3);

                // Fluctuating confidence
                if (Math.random() > 0.95) obj.confidence = Math.max(80, Math.min(99, obj.confidence + (Math.random() > 0.5 ? 1 : -1)));

                const baseX = (obj.rx * width) + jitterX;
                const baseY = (obj.ry * height) + jitterY;
                const baseW = obj.rw * width;
                const baseH = obj.rh * height;

                // Animate expansion from center
                const minX = baseX + (baseW / 2) * (1 - scale);
                const minY = baseY + (baseH / 2) * (1 - scale);
                const boxW = baseW * scale;
                const boxH = baseH * scale;

                // 1. Cutout reveal of the bright image underneath
                ctx.save();
                ctx.beginPath();
                ctx.rect(minX, minY, boxW, boxH);
                ctx.clip();
                // Redraw the bright image inside the clip
                ctx.drawImage(bgImage, offsetX, offsetY, drawWidth, drawHeight);
                // Slight tint
                ctx.fillStyle = `rgba(255, 255, 255, ${0.05 * scale})`;
                ctx.fillRect(minX, minY, boxW, boxH);
                ctx.restore();

                // 2. Bounding Box Outlines
                ctx.strokeStyle = `rgba(255,255,255,${0.15 * scale})`;
                ctx.lineWidth = 1;
                ctx.strokeRect(minX, minY, boxW, boxH);

                if (progress > 0.5) { // Show brackets and text only when mostly expanded
                    // 3. Bold Corner Brackets
                    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
                    ctx.lineWidth = 2;
                    const len = 12 * progress;
                    // Top Left
                    ctx.beginPath(); ctx.moveTo(minX, minY + len); ctx.lineTo(minX, minY); ctx.lineTo(minX + len, minY); ctx.stroke();
                    // Top Right
                    ctx.beginPath(); ctx.moveTo(minX + boxW - len, minY); ctx.lineTo(minX + boxW, minY); ctx.lineTo(minX + boxW, minY + len); ctx.stroke();
                    // Bottom Left
                    ctx.beginPath(); ctx.moveTo(minX, minY + boxH - len); ctx.lineTo(minX, minY + boxH); ctx.lineTo(minX + len, minY + boxH); ctx.stroke();
                    // Bottom Right
                    ctx.beginPath(); ctx.moveTo(minX + boxW, minY + boxH - len); ctx.lineTo(minX + boxW, minY + boxH); ctx.lineTo(minX + boxW - len, minY + boxH); ctx.stroke();

                    // 4. Labels
                    ctx.fillStyle = `rgba(255,255,255,${progress})`;
                    ctx.font = '11px "JetBrains Mono", monospace';
                    ctx.fillText(`${obj.label} [${obj.confidence}%]`, minX, minY - 6);

                    // Processing Dot
                    if (Math.random() < 0.15) {
                        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                        ctx.beginPath();
                        ctx.arc(minX + boxW / 2, minY + boxH / 2, 2, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            });

            // Draw subtle tracking overlay
            ctx.strokeStyle = 'rgba(255,255,255,0.03)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(width / 2, 0); ctx.lineTo(width / 2, height);
            ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2);
            ctx.stroke();

            // Hardware scanline
            scanY += 2;
            if (scanY > height) scanY = 0;

            ctx.fillStyle = 'rgba(255,255,255,0.15)';
            ctx.fillRect(0, scanY, width, 1);
            const gradient = ctx.createLinearGradient(0, scanY - 40, 0, scanY);
            gradient.addColorStop(0, 'rgba(255,255,255,0)');
            gradient.addColorStop(1, 'rgba(255,255,255,0.05)');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, scanY - 40, width, 40);

        } else {
            ctx.fillStyle = '#050505';
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = '#fff';
            ctx.font = '12px "JetBrains Mono", monospace';
            ctx.fillText("SYNCING OPTICS...", width / 2 - 60, height / 2);
        }

        time++;
        requestAnimationFrame(render);
    }

    render();

    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = ((y - centerY) / centerY) * -5;
            const rotateY = ((x - centerX) / centerX) * 5;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-10px)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = `perspective(1000px) rotateX(0) rotateY(0) translateY(0)`;
        });
    });
});
