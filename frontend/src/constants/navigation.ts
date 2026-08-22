import { BookOpen, Briefcase, Layers, Mail, PartyPopper, Smile, LucideIcon } from "lucide-react";

export interface NavItem {
  id: string;
  label: string;
  icon: LucideIcon;
  prompt: string;
}

export const NAVIGATION_ITEMS: NavItem[] = [
  {
    id: "me",
    label: "Me",
    icon: Smile,
    prompt: "What's your name?",
  },
  {
    id: "projects",
    label: "Projects",
    icon: Briefcase,
    prompt: "What type of projects you did work on?",
  },
  {
    id: "skills",
    label: "Skills",
    icon: Layers,
    prompt: "What skills you do have?",
  },
  {
    id: "fun",
    label: "Fun",
    icon: PartyPopper,
    prompt: "What do you do for fun?",
  },
  {
    id: "books",
    label: "Books",
    icon: BookOpen,
    prompt: "What books you did read?",
  },
  {
    id: "contact",
    label: "Contact",
    icon: Mail,
    prompt: "Give me your contacts.",
  },
];
