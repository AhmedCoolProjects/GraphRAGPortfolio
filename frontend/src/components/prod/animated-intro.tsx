import { PointerHighlight } from "@/components/ui/pointer-highlight";

export function PointerHighlightDemo() {
  return (
    // <div className="mb-10 sm:mb-20 text-xl text-center sm:text-5xl dark:text-white text-black">
    //   I do
    //   <PointerHighlight>
    //     <span>AI Research</span>
    //   </PointerHighlight>
    //
    // </div>

    <div className=" text-xl text-center sm:text-5xl dark:text-white text-black">
      <span className="text-3xl">Hey I am Ahmed,</span>
      <br />I do
      <PointerHighlight
        rectangleClassName="bg-blue-100 dark:bg-blue-900 border-blue-300 dark:border-blue-700 leading-loose"
        pointerClassName="text-blue-500 h-3 w-3"
        containerClassName="inline-block mx-1"
      >
        <span className="relative z-10">AI Research </span>
      </PointerHighlight>
      for
      <PointerHighlight
        rectangleClassName="bg-red-100 dark:bg-red-900 border-red-300 dark:border-red-700 leading-loose"
        pointerClassName="text-red-500 h-3 w-3"
        containerClassName="inline-block mx-1"
      >
        <span className="relative z-10">Cyber Security</span>
      </PointerHighlight>
      .
    </div>
  );
}
