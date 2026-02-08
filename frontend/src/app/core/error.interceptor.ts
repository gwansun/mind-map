import {
  HttpErrorResponse,
  HttpEvent,
  HttpHandler,
  HttpInterceptor,
  HttpRequest,
} from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, retry, throwError, timer } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { ToastService } from '../shared/components/toast/toast.service';

@Injectable()
export class ErrorInterceptor implements HttpInterceptor {
  private toastService = inject(ToastService);

  intercept(request: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    return next.handle(request).pipe(
      // Retry on transient failures (network errors, 503)
      retry({
        count: 2,
        delay: (error, retryCount) => {
          // Only retry on network errors or 503
          if (error instanceof HttpErrorResponse) {
            if (error.status === 0 || error.status === 503) {
              return timer(1000 * retryCount); // Exponential backoff
            }
          }
          return throwError(() => error);
        },
      }),
      catchError((error: HttpErrorResponse) => {
        this.handleError(error);
        return throwError(() => error);
      })
    );
  }

  private handleError(error: HttpErrorResponse): void {
    let message: string;

    if (error.status === 0) {
      // Network error
      message = 'Unable to connect to server. Please check your connection.';
    } else if (error.status === 404) {
      message = 'The requested resource was not found.';
    } else if (error.status === 422) {
      // Validation error
      message = error.error?.detail?.[0]?.msg || 'Invalid request data.';
    } else if (error.status >= 500) {
      message = 'Server error. Please try again later.';
    } else if (error.error?.detail) {
      message = error.error.detail;
    } else {
      message = 'An unexpected error occurred.';
    }

    this.toastService.error(message);
  }
}
