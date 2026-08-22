"use client";

// Self-contained canvas force-directed graph. No external graph library —
// a small spring/repulsion simulation on requestAnimationFrame. Renders the
// full knowledge graph as an ambient backdrop and highlights the subgraph the
// agent traversed for the latest answer (dimming everything else).

import { useEffect, useRef } from "react";
import {
  FullGraph,
  GraphPath,
  typeColor,
} from "@/lib/graph";

interface SimNode {
  id: string;
  label: string;
  type: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface Props {
  graph: FullGraph;
  highlight?: GraphPath | null;
  className?: string;
}

export function KnowledgeGraph({ graph, highlight, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const nodesRef = useRef<Map<string, SimNode>>(new Map());
  const hoverRef = useRef<string | null>(null);
  const highlightRef = useRef<GraphPath | null | undefined>(highlight);
  const rafRef = useRef<number>(0);
  const darkRef = useRef<boolean>(false);

  // Keep the latest highlight available to the render loop without restarting it.
  useEffect(() => {
    highlightRef.current = highlight;
  }, [highlight]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // ---- build sim nodes (preserve positions across graph identity) --------
    const sim = nodesRef.current;
    const center = { x: 0, y: 0 };
    graph.nodes.forEach((n, i) => {
      if (!sim.has(n.id)) {
        const angle = (i / graph.nodes.length) * Math.PI * 2;
        const r = 120 + Math.random() * 80;
        sim.set(n.id, {
          id: n.id,
          label: n.label,
          type: n.type,
          x: Math.cos(angle) * r,
          y: Math.sin(angle) * r,
          vx: 0,
          vy: 0,
        });
      }
    });
    const adjacency = graph.edges
      .map((e) => [sim.get(e.source), sim.get(e.target)] as const)
      .filter(([a, b]) => a && b) as [SimNode, SimNode][];

    // ---- sizing / DPR ------------------------------------------------------
    let width = 0;
    let height = 0;
    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      width = rect.width;
      height = rect.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    // ---- dark-mode tracking ------------------------------------------------
    const updateDark = () =>
      (darkRef.current = document.documentElement.classList.contains("dark"));
    updateDark();
    const mo = new MutationObserver(updateDark);
    mo.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    // ---- pointer -----------------------------------------------------------
    const onMove = (ev: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left - width / 2;
      const my = ev.clientY - rect.top - height / 2;
      let best: string | null = null;
      let bestD = 14 * 14;
      sim.forEach((n) => {
        const dx = n.x - mx;
        const dy = n.y - my;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          best = n.id;
        }
      });
      hoverRef.current = best;
      canvas.style.cursor = best ? "pointer" : "default";
    };
    const onLeave = () => (hoverRef.current = null);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    // ---- simulation step ---------------------------------------------------
    const REPULSION = 1400;
    const SPRING = 0.02;
    const SPRING_LEN = 70;
    const CENTER_PULL = 0.012;
    const DAMP = 0.82;

    const step = () => {
      const arr = Array.from(sim.values());
      // repulsion (O(n^2), n≈80 — trivial)
      for (let i = 0; i < arr.length; i++) {
        const a = arr[i];
        for (let j = i + 1; j < arr.length; j++) {
          const b = arr[j];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 0.01) {
            dx = Math.random() - 0.5;
            dy = Math.random() - 0.5;
            d2 = 0.01;
          }
          const f = REPULSION / d2;
          const d = Math.sqrt(d2);
          const fx = (dx / d) * f;
          const fy = (dy / d) * f;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }
      }
      // springs along edges
      for (const [a, b] of adjacency) {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - SPRING_LEN) * SPRING;
        const fx = (dx / d) * f;
        const fy = (dy / d) * f;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }
      // centering + integrate
      for (const n of arr) {
        n.vx -= n.x * CENTER_PULL;
        n.vy -= n.y * CENTER_PULL;
        n.vx *= DAMP;
        n.vy *= DAMP;
        n.x += n.vx;
        n.y += n.vy;
      }
    };

    // ---- render ------------------------------------------------------------
    const draw = () => {
      const dark = darkRef.current;
      const hl = highlightRef.current;
      const hlNodes = hl ? new Set(hl.nodes.map((n) => n.id)) : null;
      const seeds = hl ? new Set(hl.nodes.filter((n) => n.seed).map((n) => n.id)) : null;
      const hlEdges = hl
        ? new Set(hl.edges.map((e) => `${e.source}->${e.target}`))
        : null;
      const hover = hoverRef.current;

      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.translate(width / 2, height / 2);

      const dimEdge = dark ? "rgba(148,163,184,0.10)" : "rgba(100,116,139,0.12)";
      const dimNode = dark ? "rgba(148,163,184,0.22)" : "rgba(100,116,139,0.22)";

      // edges
      for (const e of graph.edges) {
        const a = sim.get(e.source);
        const b = sim.get(e.target);
        if (!a || !b) continue;
        const on =
          !hlEdges ||
          hlEdges.has(`${e.source}->${e.target}`) ||
          hlEdges.has(`${e.target}->${e.source}`);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        if (hl && on) {
          ctx.strokeStyle = dark ? "rgba(96,165,250,0.55)" : "rgba(37,99,235,0.45)";
          ctx.lineWidth = 1.6;
        } else {
          ctx.strokeStyle = dimEdge;
          ctx.lineWidth = 1;
        }
        ctx.stroke();
      }

      // nodes
      sim.forEach((n) => {
        const on = !hlNodes || hlNodes.has(n.id);
        const isSeed = seeds?.has(n.id);
        const isHover = hover === n.id;
        const base = typeColor(n.type);
        const r = isSeed ? 7 : n.type === "person" ? 7 : 4.5;

        if (hl && on) {
          // glow for active nodes
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + (isSeed ? 6 : 3), 0, Math.PI * 2);
          ctx.fillStyle = base + "33";
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, isHover ? r + 2 : r, 0, Math.PI * 2);
        ctx.fillStyle = !hl || on ? base : dimNode;
        ctx.fill();
        if (isSeed || n.type === "person") {
          ctx.lineWidth = 2;
          ctx.strokeStyle = dark ? "#fff" : "#0a0a0a";
          ctx.stroke();
        }

        // labels: show for active/seed/hovered/person
        const showLabel =
          isHover || isSeed || n.type === "person" || (hl && on && hlNodes!.size <= 12);
        if (showLabel) {
          ctx.font = `${isSeed || n.type === "person" ? 600 : 400} 11px ui-sans-serif, system-ui`;
          ctx.fillStyle = dark ? "rgba(244,244,245,0.95)" : "rgba(24,24,27,0.95)";
          ctx.textAlign = "center";
          ctx.fillText(n.label, n.x, n.y - r - 5);
        }
      });

      ctx.restore();
    };

    let frames = 0;
    const loop = () => {
      // The layout settles quickly; keep simulating while highlighting or for
      // the first ~600 frames, then idle-render (cheap) so hover still works.
      if (frames < 600 || highlightRef.current) step();
      frames++;
      draw();
      rafRef.current = requestAnimationFrame(loop);
    };
    loop();

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      mo.disconnect();
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  }, [graph]);

  return (
    <div ref={wrapRef} className={className} style={{ width: "100%", height: "100%" }}>
      <canvas ref={canvasRef} />
    </div>
  );
}
