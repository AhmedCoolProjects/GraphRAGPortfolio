export interface PersonalInfo {
  name: string;
  age?: number;
  location: string;
  title: string;
  bio: string;
  tagline: string;
  tags: string[];
}

export interface Education {
  degree: string;
  institution: string;
  year: string;
  field?: string;
  description?: string;
}

export interface Project {
  name: string;
  description: string;
  technologies: string[];
  link?: string;
  status?: string;
}

export interface Skill {
  category: string;
  items: string[];
}

export interface Book {
  title: string;
  author: string;
  category?: string;
  recommendation?: string;
}

export interface ContactInfo {
  email: string;
  linkedin?: string;
  github?: string;
  twitter?: string;
  website?: string;
}

export interface KnowledgeBase {
  personal: PersonalInfo;
  education: Education[];
  projects: Project[];
  skills: Skill[];
  interests: string[];
  books: Book[];
  contact: ContactInfo;
  funFacts: string[];
}

// Your personal knowledge base
export const KNOWLEDGE_BASE: KnowledgeBase = {
  personal: {
    name: "Ahmed Bargady",
    age: 28,
    location: "Morocco",
    title: "PhD Student in AI & Cybersecurity",
    tagline: "Blending AI with Cybersecurity to create innovative solutions",
    bio: "I'm a PhD student at UM6P, passionate about artificial intelligence and cybersecurity. I believe in creating technology that makes a difference.",
    tags: ["AI", "Cybersecurity", "Research", "Innovation", "Open Source"],
  },

  education: [
    {
      degree: "PhD in AI & Cybersecurity",
      institution: "Mohammed VI Polytechnic University (UM6P)",
      year: "2023 - Present",
      field: "Artificial Intelligence & Cybersecurity",
      description: "Researching advanced AI techniques for cybersecurity applications",
    },
    {
      degree: "Master's Degree",
      institution: "Your University",
      year: "2020 - 2022",
      field: "Computer Science",
    },
    // Add more education entries
  ],

  projects: [
    {
      name: "AI Security Framework",
      description: "A comprehensive framework for detecting and preventing cyber threats using machine learning",
      technologies: ["Python", "TensorFlow", "PyTorch", "Docker"],
      status: "In Progress",
    },
    {
      name: "Portfolio Website",
      description: "Interactive portfolio with AI-powered chat using Google Gemini",
      technologies: ["Next.js", "TypeScript", "Tailwind CSS", "Gemini AI"],
      status: "Active",
    },
    // Add more projects
  ],

  skills: [
    {
      category: "Programming Languages",
      items: ["Python", "JavaScript", "TypeScript", "Java", "C++"],
    },
    {
      category: "AI & Machine Learning",
      items: ["TensorFlow", "PyTorch", "Scikit-learn", "NLP", "Computer Vision"],
    },
    {
      category: "Web Development",
      items: ["React", "Next.js", "Node.js", "Tailwind CSS"],
    },
    {
      category: "Cybersecurity",
      items: ["Penetration Testing", "Network Security", "Cryptography", "Threat Detection"],
    },
    {
      category: "Tools & Technologies",
      items: ["Git", "Docker", "Linux", "AWS", "MongoDB"],
    },
  ],

  interests: [
    "Artificial Intelligence",
    "Cybersecurity Research",
    "Open Source Contribution",
    "Technology Innovation",
    "Reading Tech Books",
    "Problem Solving",
  ],

  books: [
    {
      title: "Deep Learning",
      author: "Ian Goodfellow",
      category: "AI/ML",
      recommendation: "Essential for understanding deep learning fundamentals",
    },
    {
      title: "The Pragmatic Programmer",
      author: "Andrew Hunt & David Thomas",
      category: "Software Engineering",
      recommendation: "Great book for professional software development",
    },
    // Add more books
  ],

  contact: {
    email: "ahmed.bargady@um6p.ma",
    linkedin: "linkedin.com/in/ahmed-bargady",
    github: "github.com/AhmedCoolProjects",
    twitter: "@ahmedbargady",
    website: "ahmedbargady.com",
  },

  funFacts: [
    "I started coding when I was 15 years old",
    "I love solving complex problems with AI",
    "I'm passionate about teaching and mentoring",
    "I enjoy contributing to open-source projects",
    "I believe technology should be accessible to everyone",
  ],
};

// Function to generate context for Gemini
export function generateKnowledgeContext(): string {
  const kb = KNOWLEDGE_BASE;
  
  return `
You are an AI assistant representing ${kb.personal.name}, a ${kb.personal.title} from ${kb.personal.location}.

PERSONAL INFORMATION:
- Name: ${kb.personal.name}
- Title: ${kb.personal.title}
- Location: ${kb.personal.location}
- Bio: ${kb.personal.bio}
- Tagline: ${kb.personal.tagline}
- Tags: ${kb.personal.tags.join(", ")}

EDUCATION:
${kb.education.map(edu => `- ${edu.degree} at ${edu.institution} (${edu.year})${edu.description ? ': ' + edu.description : ''}`).join('\n')}

PROJECTS:
${kb.projects.map(proj => `- ${proj.name}: ${proj.description}\n  Technologies: ${proj.technologies.join(", ")}\n  Status: ${proj.status || 'Completed'}`).join('\n\n')}

SKILLS:
${kb.skills.map(skill => `- ${skill.category}: ${skill.items.join(", ")}`).join('\n')}

INTERESTS:
${kb.interests.map(interest => `- ${interest}`).join('\n')}

BOOKS I RECOMMEND:
${kb.books.map(book => `- "${book.title}" by ${book.author}${book.recommendation ? ' - ' + book.recommendation : ''}`).join('\n')}

CONTACT:
- Email: ${kb.contact.email}
${kb.contact.linkedin ? `- LinkedIn: ${kb.contact.linkedin}` : ''}
${kb.contact.github ? `- GitHub: ${kb.contact.github}` : ''}
${kb.contact.twitter ? `- Twitter: ${kb.contact.twitter}` : ''}

FUN FACTS:
${kb.funFacts.map(fact => `- ${fact}`).join('\n')}

When answering questions:
1. Respond as if you are ${kb.personal.name}, using first person ("I", "my", etc.)
2. Be professional but friendly and approachable
3. Use the information provided above to answer questions about education, projects, skills, interests, books, and contact
4. If asked about something not in this knowledge base, politely indicate that you don't have that specific information
5. When discussing projects or skills, be enthusiastic and detailed
6. Format your responses nicely with markdown for better readability
`;
}
