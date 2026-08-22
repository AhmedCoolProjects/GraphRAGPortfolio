"use client";

import { PlaceholdersAndVanishInput } from "@comp/ui/placeholders-and-vanish-input";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useStore } from "@/store/useStore";

export function PlaceholdersAndVanishInputDemo() {
  const router = useRouter();
  const { setInitialMessage } = useStore();
  const [inputValue, setInputValue] = useState("");

  const placeholders = [
    "Who are you?",
    "What your last degree?",
    "What's your research about?",
    "What are your career goals?",
    "What skills do you want to learn?",
    "What do you do in your free time?",
  ];

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };
  
  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (inputValue.trim()) {
      // Store the message in global state
      setInitialMessage(inputValue.trim());
      // Redirect to chat page
      router.push("/chat");
    }
  };
  
  return (
    <PlaceholdersAndVanishInput
      placeholders={placeholders}
      onChange={handleChange}
      onSubmit={onSubmit}
    />
  );
}
