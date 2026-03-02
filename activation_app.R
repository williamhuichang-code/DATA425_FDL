library(shiny)
library(plotly)

# ---- grid ----
x1 <- seq(-3, 3, length.out = 80)
x2 <- seq(-3, 3, length.out = 80)
grid <- expand.grid(x1 = x1, x2 = x2)
relu <- function(z) pmax(0, z)

# ---- compute surfaces ----
compute_surfaces <- function(n, seed = 42) {
  set.seed(seed)
  
  W <- matrix(runif(n * 2, -1, 1), nrow = n, ncol = 2)
  b <- runif(n, -0.5, 0.5)
  
  # pre-activations: each column = wᵢ·x + bᵢ
  pre <- matrix(0, nrow = nrow(grid), ncol = n)
  for (i in seq_len(n)) {
    pre[, i] <- W[i,1]*grid$x1 + W[i,2]*grid$x2 + b[i]
  }
  
  hidden <- apply(pre, 2, relu)
  
  # blue:  sum of linear pre-activations (what network would be WITHOUT ReLU)
  # red:   sum of ReLU outputs           (actual network output)
  # green: individual ReLU neurons (each column of hidden)
  list(
    z_linear = matrix(rowSums(pre),    nrow = 80),
    z_relu   = matrix(rowSums(hidden), nrow = 80),
    hidden   = hidden,
    n        = n,
    W        = W,
    b        = b
  )
}

# ---- UI ----
ui <- navbarPage(
  title = "Neural Network Explorer",
  
  tabPanel("Activation",
           sidebarLayout(
             
             sidebarPanel(
               width = 3,
               
               h4("🎥 Camera"),
               actionButton("btn_rotate", "▶ Rotate", class = "btn-success btn-block"),
               tags$br(),
               actionButton("btn_pause",  "⏸ Pause",  class = "btn-warning btn-block"),
               
               tags$hr(),
               
               h4("🧠 Architecture"),
               sliderInput("n_neurons",
                           label   = "Hidden Layer Width (n):",
                           min     = 1, max = 20, value = 1, step = 1),
               actionButton("btn_reseed", "🎲 New Random Weights", class = "btn-default btn-block"),
               div(style = "font-size:12px; color:#666; margin-top:6px; margin-bottom:4px;",
                   textOutput("seed_display", inline = TRUE)),
               
               tags$hr(),
               
               h4("📐 Planes"),
               checkboxInput("show_linear",
                             tags$span(style = "color:#4444ff; font-weight:bold;", "Linear Plane"),
                             value = TRUE),
               checkboxInput("show_neurons",
                             tags$span(style = "color:#22aa22; font-weight:bold;", "Hidden Neurons"),
                             value = FALSE),
               checkboxInput("show_relu",
                             tags$span(style = "color:#cc2222; font-weight:bold;", "Network Output"),
                             value = FALSE),
               
               tags$hr(),
               
               h4("ℹ️ Info"),
               conditionalPanel("input.show_linear",
                                div(class = "alert alert-info", style = "font-size:12px;",
                                    "The ", strong("linear plane"), " shows Σ(wᵢ·x + bᵢ) — what the network
               would output with no activation function.")
               ),
               conditionalPanel("input.show_neurons",
                                div(class = "alert alert-success", style = "font-size:12px;",
                                    strong("Each green surface"), " is one neuron: ReLU(wᵢ·x + bᵢ).
               With n=1, this matches the network output exactly.")
               ),
               conditionalPanel("input.show_relu",
                                div(class = "alert alert-danger", style = "font-size:12px;",
                                    "The ", strong("network output"), " = Σ ReLU(wᵢ·x + bᵢ).
               With n=1 it equals the single green neuron.
               More neurons → richer piecewise surface.")
               ),
               
               tags$hr(),
               h4("⚙️ Weights"),
               uiOutput("weight_table")
             ),
             
             mainPanel(
               width = 9,
               plotlyOutput("plot3d", height = "650px")
             )
           )
  ),
  
  tags$script(HTML("
    var rotateTimer = null;
    var angle = 0;

    function doRotate() {
      angle += 0.5;
      var rad = angle * Math.PI / 180;
      var plotEl = document.getElementById('plot3d');
      if (plotEl && plotEl._fullLayout) {
        Plotly.relayout(plotEl, {
          'scene.camera.eye': {
            x: 1.8 * Math.cos(rad),
            y: 1.8 * Math.sin(rad),
            z: 0.8
          }
        });
      }
    }

    $(document).on('click', '#btn_rotate', function() {
      if (!rotateTimer) {
        rotateTimer = setInterval(doRotate, 30);
      }
    });

    $(document).on('click', '#btn_pause', function() {
      clearInterval(rotateTimer);
      rotateTimer = null;
    });
  "))
)

# ---- Server ----
server <- function(input, output, session) {
  
  seed_val <- reactiveVal(42)
  
  observeEvent(input$btn_reseed, {
    seed_val(sample(1:9999, 1))
  })
  
  output$seed_display <- renderText({
    paste("seed =", seed_val())
  })
  
  surfaces <- reactive({
    compute_surfaces(input$n_neurons, seed = seed_val())
  })
  
  output$plot3d <- renderPlotly({
    s <- surfaces()
    n <- s$n
    p <- plot_ly()
    
    if (input$show_linear) {
      p <- p %>% add_surface(
        x = x1, y = x2, z = s$z_linear,
        opacity    = 0.5,
        colorscale = list(c(0,1), c("blue","blue")),
        name       = "Linear Plane",
        showscale  = FALSE
      )
    }
    
    if (input$show_neurons) {
      greens <- colorRampPalette(c("#00cc44", "#004d1a"))(max(n, 2))
      for (i in seq_len(n)) {
        p <- p %>% add_surface(
          x = x1, y = x2,
          z = matrix(s$hidden[, i], nrow = 80),
          opacity    = max(0.15, 0.5 / n),
          colorscale = list(c(0,1), c(greens[i], greens[i])),
          name       = paste0("Neuron ", i),
          showscale  = FALSE
        )
      }
    }
    
    if (input$show_relu) {
      p <- p %>% add_surface(
        x = x1, y = x2, z = s$z_relu,
        opacity    = 0.65,
        colorscale = list(c(0,1), c("red","red")),
        name       = "Network Output",
        showscale  = FALSE
      )
    }
    
    if (!input$show_linear && !input$show_neurons && !input$show_relu) {
      p <- p %>% add_annotations(
        text      = "Select at least one plane from the sidebar",
        x = 0.5, y = 0.5, xref = "paper", yref = "paper",
        showarrow = FALSE, font = list(size = 16)
      )
    }
    
    p %>% layout(
      scene = list(
        xaxis  = list(title = "x1"),
        yaxis  = list(title = "x2"),
        zaxis  = list(title = "f(x)"),
        camera = list(eye = list(x = 1.8, y = 0, z = 0.8))
      ),
      legend = list(x = 0.01, y = 0.99),
      margin = list(l = 0, r = 0, t = 40, b = 0),
      title  = paste0("Hidden Layer: ", n, " neuron", ifelse(n == 1, "", "s"),
                      " | Output = Σ ReLU(wᵢ·x + bᵢ)")
    )
  })
  
  output$weight_table <- renderUI({
    s <- surfaces()
    rows <- lapply(seq_len(s$n), function(i) {
      tags$tr(
        tags$td(paste0("N", i)),
        tags$td(round(s$W[i,1], 2)),
        tags$td(round(s$W[i,2], 2)),
        tags$td(round(s$b[i],   2))
      )
    })
    tags$table(
      class = "table table-condensed table-bordered",
      style = "font-size:11px;",
      tags$thead(tags$tr(
        tags$th(""), tags$th("w1"), tags$th("w2"), tags$th("b")
      )),
      tags$tbody(rows)
    )
  })
}

shinyApp(ui, server)