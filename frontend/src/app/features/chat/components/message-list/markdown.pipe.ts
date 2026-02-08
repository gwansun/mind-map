import { Pipe, PipeTransform } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { marked } from 'marked';

@Pipe({
  name: 'markdown',
  standalone: true,
})
export class MarkdownPipe implements PipeTransform {
  constructor(private sanitizer: DomSanitizer) {
    // Configure marked for safe output
    marked.setOptions({
      breaks: true,
      gfm: true,
    });
  }

  transform(value: string | null | undefined): SafeHtml {
    if (!value) return '';

    try {
      const html = marked.parse(value, { async: false }) as string;
      return this.sanitizer.bypassSecurityTrustHtml(html);
    } catch {
      return value;
    }
  }
}
