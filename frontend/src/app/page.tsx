'use client'
import { motion } from "motion/react";
import { AnimatedProfileImage } from "@/components/ui/animated-profile-image";
import { AnimatedChatInput } from "@/components/prod/animated-chat-input";
import { NAVIGATION_ITEMS } from "@/constants/navigation";
import { KNOWLEDGE_BASE } from "@/constants/knowledge-base";
import { useRouter } from "next/navigation";
import { useStore } from '@/store/useStore'
import { Github, Linkedin, Mail, ArrowDown, Brain, Shield, Code2, ArrowUpRight } from "lucide-react";
import { PointerHighlight } from "@/components/ui/pointer-highlight";

const SOCIALS = {
  github: `https://${KNOWLEDGE_BASE.contact.github}`,
  linkedin: `https://${KNOWLEDGE_BASE.contact.linkedin}`,
  email: KNOWLEDGE_BASE.contact.email,
};

const HIGHLIGHTS = [
  {
    icon: Brain,
    title: "AI Research",
    body: "PhD work on AI-driven intrusion detection — deep learning, NLP, and applied ML at UM6P.",
    accent: "text-blue-500 dark:text-blue-400",
  },
  {
    icon: Shield,
    title: "Cybersecurity",
    body: "Threat detection, network security, and building defensive systems that actually ship.",
    accent: "text-red-500 dark:text-red-400",
  },
  {
    icon: Code2,
    title: "Engineering",
    body: "Full-stack with Next.js, Python, and TypeScript. Open-source contributor.",
    accent: "text-emerald-500 dark:text-emerald-400",
  },
];

export default function Home() {
  const router = useRouter();
  const { setInitialMessage } = useStore();

  const handleNavClick = (prompt: string) => {
    setInitialMessage(prompt);
    router.push("/chat");
  };

  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 px-6 py-4 backdrop-blur-sm bg-white/70 dark:bg-zinc-950/70">
        <div className="max-w-5xl mx-auto flex justify-between items-center">
          <a href="#top" className="text-sm font-medium tracking-tight">AB.</a>
          <div className="flex items-center gap-4">
            <a
              href={SOCIALS.github}
              aria-label="GitHub"
              className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Github size={18} />
            </a>
            <a
              href={SOCIALS.linkedin}
              aria-label="LinkedIn"
              className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Linkedin size={18} />
            </a>
            <a
              href={`mailto:${SOCIALS.email}`}
              aria-label="Email"
              className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
            >
              <Mail size={18} />
            </a>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <main id="top" className="min-h-screen flex flex-col items-center justify-center px-6 pt-24 pb-16 sm:pt-28">
        <div className="max-w-3xl mx-auto w-full text-center">
          {/* Profile */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="mb-8 flex justify-center"
          >
            <div className="w-28 h-28 sm:w-32 sm:h-32">
              <AnimatedProfileImage
                src="/me.png"
                alt="Ahmed Bargady"
                width={128}
                height={128}
                className="rounded-full"
              />
            </div>
          </motion.div>

          {/* Name */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="text-3xl sm:text-5xl font-semibold tracking-tight mb-3"
          >
            Ahmed BARGADY
          </motion.h1>

          {/* Role */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
            className="text-base sm:text-lg text-zinc-500 dark:text-zinc-400 mb-3 flex flex-wrap justify-center items-center gap-x-2 gap-y-1"
          >
            <span>PhD Student in</span>
            <PointerHighlight
              rectangleClassName="bg-blue-50 dark:bg-blue-950/30 border-blue-300 dark:border-blue-700"
              pointerClassName="text-blue-500 h-3 w-3"
              containerClassName="inline-block"
            >
              <span className="relative z-10 font-semibold text-blue-600 dark:text-blue-400 px-1">
                AI
              </span>
            </PointerHighlight>
            <span>&</span>
            <PointerHighlight
              rectangleClassName="bg-red-50 dark:bg-red-950/30 border-red-300 dark:border-red-700"
              pointerClassName="text-red-500 h-3 w-3"
              containerClassName="inline-block"
            >
              <span className="relative z-10 font-semibold text-red-600 dark:text-red-400 px-1">
                Cybersecurity
              </span>
            </PointerHighlight>
          </motion.p>

          {/* Tagline */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3, ease: "easeOut" }}
            className="text-sm sm:text-base text-zinc-600 dark:text-zinc-500 max-w-xl mx-auto mb-10 leading-relaxed"
          >
            Researching AI-driven security at UM6P. Passionate about machine learning,
            intrusion detection, and building intelligent systems.
          </motion.p>

          {/* Primary CTA: Chat input */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4, ease: "easeOut" }}
            className="mb-6"
          >
            <AnimatedChatInput />
          </motion.div>

          {/* Quick prompts */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.55, ease: "easeOut" }}
            className="mt-2"
          >
            <p className="text-xs uppercase tracking-wider text-zinc-400 dark:text-zinc-600 mb-3">
              Or try a quick prompt
            </p>
            <div className="flex flex-wrap justify-center gap-2 sm:gap-3">
              {NAVIGATION_ITEMS.map((item, index) => {
                const colors = [
                  { text: "text-blue-600 dark:text-blue-400", icon: "text-blue-500 dark:text-blue-400" },
                  { text: "text-emerald-600 dark:text-emerald-400", icon: "text-emerald-500 dark:text-emerald-400" },
                  { text: "text-violet-600 dark:text-violet-400", icon: "text-violet-500 dark:text-violet-400" },
                  { text: "text-amber-600 dark:text-amber-400", icon: "text-amber-500 dark:text-amber-400" },
                  { text: "text-rose-600 dark:text-rose-400", icon: "text-rose-500 dark:text-rose-400" },
                  { text: "text-cyan-600 dark:text-cyan-400", icon: "text-cyan-500 dark:text-cyan-400" },
                ];
                const color = colors[index % colors.length];

                return (
                  <motion.button
                    key={item.id}
                    onClick={() => handleNavClick(item.prompt)}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.6 + index * 0.04 }}
                    whileHover={{ y: -2 }}
                    className="group flex items-center gap-2 px-4 py-2 rounded-full
                      bg-zinc-100 dark:bg-zinc-900
                      hover:bg-zinc-200 dark:hover:bg-zinc-800
                      text-xs sm:text-sm font-medium
                      transition-colors duration-200"
                  >
                    <item.icon size={14} className={color.icon} />
                    <span className={color.text}>{item.label}</span>
                  </motion.button>
                );
              })}
            </div>
          </motion.div>

          {/* Scroll affordance */}
          <motion.a
            href="#about"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 1.2 }}
            className="mt-14 inline-flex flex-col items-center gap-1 text-xs text-zinc-400 dark:text-zinc-600 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            aria-label="Scroll to about section"
          >
            <span>More about me</span>
            <motion.span
              animate={{ y: [0, 4, 0] }}
              transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
            >
              <ArrowDown size={14} />
            </motion.span>
          </motion.a>
        </div>
      </main>

      {/* About + Highlights */}
      <section id="about" className="px-6 py-20 sm:py-24 border-t border-zinc-100 dark:border-zinc-900">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
            className="max-w-2xl mb-14"
          >
            <p className="text-xs uppercase tracking-wider text-zinc-400 dark:text-zinc-600 mb-3">
              About
            </p>
            <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-4">
              Building at the edge of AI and security.
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 leading-relaxed">
              I&apos;m a PhD student at Mohammed VI Polytechnic University working on
              applied machine learning for cybersecurity. I split my time between research,
              engineering, and shipping open-source projects.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
            {HIGHLIGHTS.map((h, i) => (
              <motion.div
                key={h.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: i * 0.08 }}
                className="group p-5 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors"
              >
                <h.icon size={20} className={`${h.accent} mb-3`} />
                <h3 className="font-semibold mb-1.5 text-sm">{h.title}</h3>
                <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  {h.body}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact CTA */}
      <section className="px-6 pb-20 sm:pb-28">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6 }}
            className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-8 sm:p-10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-6"
          >
            <div>
              <h3 className="text-xl sm:text-2xl font-semibold tracking-tight mb-2">
                Want to chat?
              </h3>
              <p className="text-sm text-zinc-600 dark:text-zinc-400 max-w-md">
                Reach out about research, collaboration, or just to say hi.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <a
                href={`mailto:${SOCIALS.email}`}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 text-sm font-medium hover:opacity-90 transition-opacity"
              >
                <Mail size={16} />
                Email me
              </a>
              <a
                href={SOCIALS.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full border border-zinc-200 dark:border-zinc-800 text-sm font-medium hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
              >
                <Linkedin size={16} />
                LinkedIn
                <ArrowUpRight size={14} className="text-zinc-400" />
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-6 border-t border-zinc-100 dark:border-zinc-900">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-2 text-xs text-zinc-400 dark:text-zinc-600">
          <span>© 2026 Ahmed Bargady</span>
          <span>Casablanca, Morocco</span>
        </div>
      </footer>
    </div>
  );
}
