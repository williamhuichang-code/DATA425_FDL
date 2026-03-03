library(shiny)
library(plotly)

# ---- precompute data for Activation tab ----
x1 <- seq(-3, 3, length.out = 100)
x2 <- seq(-3, 3, length.out = 100)
grid <- expand.grid(x1 = x1, x2 = x2)

relu <- function(z) pmax(0, z)
z_linear <- 1*grid$x1 - 1*grid$x2
z1 <- relu( 1*grid$x1 - 1*grid$x2 )
z2 <- relu(-1*grid$x1 - 1*grid$x2 + 0.5)
z_relu <- 1*z1 - 1*z2

z_linear_mat <- matrix(z_linear, nrow = 100)
z_relu_mat   <- matrix(z_relu,   nrow = 100)
z1_mat       <- matrix(z1, 100)

# ---- Helper: forward pass for Q2 network ----
# Network: 2 inputs -> hidden layer 1 (3 nodes) -> hidden layer 2 (2 nodes) -> output
# All activations are linear (identity)
#
# Weights read from diagram:
#   Layer 1: W1 (3x2), b1 (3x1)
#     h1^(1): x1*1  + x2*0  + b=0
#     h2^(1): x1*(-1) + x2*2 + b=1
#     h3^(1): x1*0  + x2*1  + b=-1
#   Layer 2: W2 (2x3), b2 (2x1)
#     h1^(2): h1^(1)*1 + h2^(1)*0 + h3^(1)*2  + b=0
#     h2^(2): h1^(1)*(-1) + h2^(1)*1 + h3^(1)*0 + b=1
#   Output: W3 (1x2), b3
#     y_hat: h1^(2)*1 + h2^(2)*(-2) + b=0.5

forward_q2 <- function(x_in, W1, b1, W2, b2, w3, b3) {
  h1 <- W1 %*% x_in + b1   # 3x1
  h2 <- W2 %*% h1  + b2    # 2x1
  y  <- sum(w3 * h2) + b3
  list(h1 = h1, h2 = h2, y = y)
}

# Default weights from diagram
default_W1 <- matrix(c(1, 0,
                       -1, 2,
                       0, 1), nrow = 3, byrow = TRUE)
default_b1 <- c(0, 1, -1)

default_W2 <- matrix(c(1, 0, 2,
                       -1, 1, 0), nrow = 2, byrow = TRUE)
default_b2 <- c(0, 1)

default_w3 <- c(1, -2)
default_b3 <- 0.5

# Compute collapsed perceptron weights analytically
# y = w3 * (W2 * (W1*x + b1) + b2) + b3
#   = w3 * W2 * W1 * x  +  w3 * W2 * b1  +  w3 * b2  +  b3
collapsed_weights <- function(W1, b1, W2, b2, w3, b3) {
  Wc <- as.vector(w3 %*% W2 %*% W1)
  bc <- as.vector(w3 %*% (W2 %*% b1 + b2)) + b3
  list(w = Wc, b = bc)
}

# ---- UI ----
ui <- navbarPage(
  title = "Neural Network Explorer",
  
  # ── Tab 1: Activation ──────────────────────────────────────────────────────
  tabPanel("Activation",
           sidebarLayout(
             sidebarPanel(
               width = 3,
               h4("🎥 Camera"),
               actionButton("btn_rotate", "▶ Rotate", class = "btn-success btn-block"),
               tags$br(),
               actionButton("btn_pause",  "⏸ Pause",  class = "btn-warning btn-block"),
               tags$hr(),
               h4("📐 Planes"),
               checkboxInput("show_linear",
                             tags$span(style="color:#4444ff;font-weight:bold;","Linear Plane"),
                             value = TRUE),
               checkboxInput("show_neuron",
                             tags$span(style="color:#22aa22;font-weight:bold;","Hidden Neuron 1"),
                             value = FALSE),
               checkboxInput("show_relu",
                             tags$span(style="color:#cc2222;font-weight:bold;","ReLU Surface"),
                             value = FALSE),
               tags$hr(),
               h4("ℹ️ Info"),
               conditionalPanel("input.show_linear",
                                div(class="alert alert-info",style="font-size:12px;",
                                    "The ",strong("linear plane")," shows a standard linear transformation: f(x) = x1 - x2")),
               conditionalPanel("input.show_neuron",
                                div(class="alert alert-success",style="font-size:12px;",
                                    strong("Hidden Neuron 1")," applies ReLU to the linear combination: ReLU(x1 - x2)")),
               conditionalPanel("input.show_relu",
                                div(class="alert alert-danger",style="font-size:12px;",
                                    "The ",strong("ReLU surface")," is the network output combining both hidden neurons."))
             ),
             mainPanel(width=9, plotlyOutput("plot3d", height="650px"))
           )
  ),
  
  # ── Tab 2: Question 2 Explorer ─────────────────────────────────────────────
  tabPanel("Question 2",
           fluidPage(
             tags$head(tags$style(HTML("
               .q2-panel { background:#f8f9fa; border-radius:8px; padding:16px; margin-bottom:12px; }
               .weight-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
               .node-box { background:white; border:2px solid #dee2e6; border-radius:6px;
                           padding:10px; margin:4px 0; font-size:13px; }
               .node-box.layer1 { border-color:#6f86d6; }
               .node-box.layer2 { border-color:#48c774; }
               .node-box.output { border-color:#ff6b6b; }
               .result-box { background:#1a1a2e; color:#e0e0ff; border-radius:8px;
                             padding:20px; font-family:monospace; font-size:14px; }
               .result-box .val { color:#7effa0; font-size:20px; font-weight:bold; }
               .result-box .match { color:#ffd700; font-size:16px; }
               .section-title { font-weight:700; font-size:15px; margin-bottom:8px;
                                padding-bottom:4px; border-bottom:2px solid #e0e0e0; }
               .badge-layer { display:inline-block; padding:2px 8px; border-radius:12px;
                              font-size:11px; font-weight:700; margin-left:6px; }
               .badge-L1 { background:#6f86d6; color:white; }
               .badge-L2 { background:#48c774; color:white; }
               .badge-out { background:#ff6b6b; color:white; }
             "))),
             
             fluidRow(
               # LEFT: Weight controls
               column(4,
                      div(class="q2-panel",
                          div(class="section-title", "⚙️ Network Weights",
                              tags$small(style="color:#888;font-weight:normal;margin-left:8px;",
                                         "(edit to explore)")),
                          
                          h6(tags$b("Layer 1"), span("(2→3)", class="badge badge-secondary"),
                             style="margin-top:12px;"),
                          tags$table(style="width:100%;font-size:12px;",
                                     tags$tr(tags$th("Node"), tags$th("w_x1"), tags$th("w_x2"), tags$th("bias")),
                                     tags$tr(
                                       tags$td(tags$span("h₁⁽¹⁾", style="color:#6f86d6;font-weight:bold;")),
                                       tags$td(numericInput("w1_11","",1,width="70px",step=0.5)),
                                       tags$td(numericInput("w1_12","",0,width="70px",step=0.5)),
                                       tags$td(numericInput("b1_1", "",0,width="70px",step=0.5))
                                     ),
                                     tags$tr(
                                       tags$td(tags$span("h₂⁽¹⁾", style="color:#6f86d6;font-weight:bold;")),
                                       tags$td(numericInput("w1_21","",-1,width="70px",step=0.5)),
                                       tags$td(numericInput("w1_22","",2,width="70px",step=0.5)),
                                       tags$td(numericInput("b1_2", "",1,width="70px",step=0.5))
                                     ),
                                     tags$tr(
                                       tags$td(tags$span("h₃⁽¹⁾", style="color:#6f86d6;font-weight:bold;")),
                                       tags$td(numericInput("w1_31","",0,width="70px",step=0.5)),
                                       tags$td(numericInput("w1_32","",1,width="70px",step=0.5)),
                                       tags$td(numericInput("b1_3","",-1,width="70px",step=0.5))
                                     )
                          ),
                          
                          h6(tags$b("Layer 2"), span("(3→2)", class="badge badge-secondary"),
                             style="margin-top:12px;"),
                          tags$table(style="width:100%;font-size:12px;",
                                     tags$tr(tags$th("Node"),tags$th("w_h1"),tags$th("w_h2"),tags$th("w_h3"),tags$th("bias")),
                                     tags$tr(
                                       tags$td(tags$span("h₁⁽²⁾", style="color:#48c774;font-weight:bold;")),
                                       tags$td(numericInput("w2_11","",1,width="60px",step=0.5)),
                                       tags$td(numericInput("w2_12","",0,width="60px",step=0.5)),
                                       tags$td(numericInput("w2_13","",2,width="60px",step=0.5)),
                                       tags$td(numericInput("b2_1","",0,width="60px",step=0.5))
                                     ),
                                     tags$tr(
                                       tags$td(tags$span("h₂⁽²⁾", style="color:#48c774;font-weight:bold;")),
                                       tags$td(numericInput("w2_21","",-1,width="60px",step=0.5)),
                                       tags$td(numericInput("w2_22","",1,width="60px",step=0.5)),
                                       tags$td(numericInput("w2_23","",0,width="60px",step=0.5)),
                                       tags$td(numericInput("b2_2","",1,width="60px",step=0.5))
                                     )
                          ),
                          
                          h6(tags$b("Output Layer"), span("(2→1)", class="badge badge-secondary"),
                             style="margin-top:12px;"),
                          tags$table(style="width:100%;font-size:12px;",
                                     tags$tr(tags$th("w_h1⁽²⁾"), tags$th("w_h2⁽²⁾"), tags$th("bias")),
                                     tags$tr(
                                       tags$td(numericInput("w3_1","",1,width="70px",step=0.5)),
                                       tags$td(numericInput("w3_2","",-2,width="70px",step=0.5)),
                                       tags$td(numericInput("b3","",0.5,width="70px",step=0.5))
                                     )
                          ),
                          
                          tags$hr(),
                          actionButton("reset_weights","↺ Reset to diagram defaults",
                                       class="btn-outline-secondary btn-sm btn-block")
                      )
               ),
               
               # MIDDLE: Test input + forward pass
               column(4,
                      div(class="q2-panel",
                          div(class="section-title", "🔢 Test Input"),
                          fluidRow(
                            column(6, numericInput("x_in1","x₁",0.5,step=0.1)),
                            column(6, numericInput("x_in2","x₂",1.0,step=0.1))
                          ),
                          actionButton("run_forward","▶ Run Forward Pass",
                                       class="btn-primary btn-block",
                                       style="margin-top:4px;"),
                          tags$hr(),
                          div(class="section-title","📊 Computation Trace"),
                          uiOutput("trace_output")
                      )
               ),
               
               # RIGHT: Results + Collapsed perceptron
               column(4,
                      div(class="q2-panel",
                          div(class="section-title","🎯 Results"),
                          uiOutput("results_panel")
                      ),
                      div(class="q2-panel",
                          div(class="section-title","⚡ Collapsed Perceptron (Part b)"),
                          p(style="font-size:12px;color:#555;",
                            "Since all activations are linear, the entire network collapses to a",
                            strong("single perceptron"), "(linear function of inputs)."),
                          uiOutput("perceptron_panel")
                      ),
                      div(class="q2-panel",
                          div(class="section-title","📖 Part (c) Verification"),
                          p(style="font-size:12px;color:#555;",
                            "Set x = [0.5, 1]ᵀ above and check both networks agree:"),
                          uiOutput("verification_panel")
                      )
               )
             )
           )
  ) # end tabPanel Q2
)

# ── Server ──────────────────────────────────────────────────────────────────
server <- function(input, output, session) {
  
  # ---------- Activation tab ----------
  output$plot3d <- renderPlotly({
    p <- plot_ly()
    if (input$show_linear) {
      p <- p %>% add_surface(x=x1,y=x2,z=z_linear_mat,opacity=0.5,
                             colorscale=list(c(0,1),c("blue","blue")),
                             name="Linear Plane",showscale=FALSE)
    }
    if (input$show_neuron) {
      p <- p %>% add_surface(x=x1,y=x2,z=z1_mat,opacity=0.5,
                             colorscale=list(c(0,1),c("green","green")),
                             name="Hidden Neuron 1",showscale=FALSE)
    }
    if (input$show_relu) {
      p <- p %>% add_surface(x=x1,y=x2,z=z_relu_mat,opacity=0.6,
                             colorscale=list(c(0,1),c("red","red")),
                             name="ReLU Surface",showscale=FALSE)
    }
    if (!input$show_linear && !input$show_neuron && !input$show_relu) {
      p <- p %>% add_annotations(text="Select at least one plane",
                                 x=0.5,y=0.5,xref="paper",yref="paper",
                                 showarrow=FALSE,font=list(size=16))
    }
    p %>% layout(scene=list(xaxis=list(title="x1"),yaxis=list(title="x2"),
                            zaxis=list(title="f(x)"),
                            camera=list(eye=list(x=1.8,y=0,z=0.8))),
                 legend=list(x=0.01,y=0.99),margin=list(l=0,r=0,t=30,b=0))
  })
  
  # ---------- Q2: reset weights ----------
  observeEvent(input$reset_weights, {
    updateNumericInput(session,"w1_11",value=1)
    updateNumericInput(session,"w1_12",value=0)
    updateNumericInput(session,"b1_1", value=0)
    updateNumericInput(session,"w1_21",value=-1)
    updateNumericInput(session,"w1_22",value=2)
    updateNumericInput(session,"b1_2", value=1)
    updateNumericInput(session,"w1_31",value=0)
    updateNumericInput(session,"w1_32",value=1)
    updateNumericInput(session,"b1_3", value=-1)
    updateNumericInput(session,"w2_11",value=1)
    updateNumericInput(session,"w2_12",value=0)
    updateNumericInput(session,"w2_13",value=2)
    updateNumericInput(session,"b2_1", value=0)
    updateNumericInput(session,"w2_21",value=-1)
    updateNumericInput(session,"w2_22",value=1)
    updateNumericInput(session,"w2_23",value=0)
    updateNumericInput(session,"b2_2", value=1)
    updateNumericInput(session,"w3_1", value=1)
    updateNumericInput(session,"w3_2", value=-2)
    updateNumericInput(session,"b3",   value=0.5)
  })
  
  # ---------- Q2: reactive forward pass ----------
  fwd <- eventReactive(input$run_forward, {
    W1 <- matrix(c(input$w1_11, input$w1_12,
                   input$w1_21, input$w1_22,
                   input$w1_31, input$w1_32), nrow=3, byrow=TRUE)
    b1 <- c(input$b1_1, input$b1_2, input$b1_3)
    W2 <- matrix(c(input$w2_11, input$w2_12, input$w2_13,
                   input$w2_21, input$w2_22, input$w2_23), nrow=2, byrow=TRUE)
    b2 <- c(input$b2_1, input$b2_2)
    w3 <- c(input$w3_1, input$w3_2)
    b3 <- input$b3
    x_in <- c(input$x_in1, input$x_in2)
    
    res <- forward_q2(x_in, W1, b1, W2, b2, w3, b3)
    cp  <- collapsed_weights(W1, b1, W2, b2, w3, b3)
    y_collapsed <- sum(cp$w * x_in) + cp$b
    
    list(W1=W1, b1=b1, W2=W2, b2=b2, w3=w3, b3=b3,
         x=x_in, h1=res$h1, h2=res$h2, y=res$y,
         cp=cp, y_collapsed=y_collapsed)
  }, ignoreNULL=FALSE)
  
  # Trace output
  output$trace_output <- renderUI({
    r <- fwd()
    if(is.null(r)) return(p("Press ▶ Run Forward Pass"))
    
    fmt <- function(v) round(v, 4)
    
    tagList(
      div(class="node-box layer1",
          tags$b("Layer 1 (h⁽¹⁾):", style="color:#6f86d6;"),
          tags$br(),
          sprintf("h₁: %s·%s + %s·%s + %s = ",
                  fmt(r$W1[1,1]), fmt(r$x[1]), fmt(r$W1[1,2]), fmt(r$x[2]), fmt(r$b1[1])),
          tags$b(fmt(r$h1[1])), tags$br(),
          sprintf("h₂: %s·%s + %s·%s + %s = ",
                  fmt(r$W1[2,1]), fmt(r$x[1]), fmt(r$W1[2,2]), fmt(r$x[2]), fmt(r$b1[2])),
          tags$b(fmt(r$h1[2])), tags$br(),
          sprintf("h₃: %s·%s + %s·%s + %s = ",
                  fmt(r$W1[3,1]), fmt(r$x[1]), fmt(r$W1[3,2]), fmt(r$x[2]), fmt(r$b1[3])),
          tags$b(fmt(r$h1[3]))
      ),
      div(class="node-box layer2",
          tags$b("Layer 2 (h⁽²⁾):", style="color:#48c774;"),
          tags$br(),
          sprintf("h₁: %s·%s + %s·%s + %s·%s + %s = ",
                  fmt(r$W2[1,1]),fmt(r$h1[1]),fmt(r$W2[1,2]),fmt(r$h1[2]),
                  fmt(r$W2[1,3]),fmt(r$h1[3]),fmt(r$b2[1])),
          tags$b(fmt(r$h2[1])), tags$br(),
          sprintf("h₂: %s·%s + %s·%s + %s·%s + %s = ",
                  fmt(r$W2[2,1]),fmt(r$h1[1]),fmt(r$W2[2,2]),fmt(r$h1[2]),
                  fmt(r$W2[2,3]),fmt(r$h1[3]),fmt(r$b2[2])),
          tags$b(fmt(r$h2[2]))
      ),
      div(class="node-box output",
          tags$b("Output ŷ:", style="color:#ff6b6b;"),
          tags$br(),
          sprintf("ŷ: %s·%s + %s·%s + %s = ",
                  fmt(r$w3[1]),fmt(r$h2[1]),fmt(r$w3[2]),fmt(r$h2[2]),fmt(r$b3)),
          tags$b(fmt(r$y), style="font-size:16px;color:#ff6b6b;")
      )
    )
  })
  
  # Results panel
  output$results_panel <- renderUI({
    r <- fwd()
    if(is.null(r)) return(p("Run forward pass to see results."))
    div(class="result-box",
        "Input: x = [", tags$b(r$x[1]), ",", tags$b(r$x[2]), "]ᵀ", tags$br(), tags$br(),
        "Original network ŷ = ", span(class="val", round(r$y, 6)), tags$br(),
        "Collapsed perceptron ŷ = ", span(class="val", round(r$y_collapsed, 6)), tags$br(), tags$br(),
        if(abs(r$y - r$y_collapsed) < 1e-9) {
          span(class="match", "✅ Both networks agree!")
        } else {
          span(style="color:#ff6b6b;", "⚠️ Mismatch — check weights")
        }
    )
  })
  
  # Collapsed perceptron panel
  output$perceptron_panel <- renderUI({
    r <- fwd()
    if(is.null(r)) return(p("Run forward pass to compute collapsed weights."))
    cp <- r$cp
    fmt <- function(v) round(v, 4)
    tagList(
      p(style="font-size:12px;",
        "Collapsed formula:", tags$br(),
        tags$code(sprintf("ŷ = %.4f·x₁ + %.4f·x₂ + %.4f", cp$w[1], cp$w[2], cp$b))),
      p(style="font-size:11px;color:#666;",
        "Derived via: w₃·W₂·W₁·x + w₃·W₂·b₁ + w₃·b₂ + b₃")
    )
  })
  
  # Verification panel (part c)
  output$verification_panel <- renderUI({
    r <- fwd()
    if(is.null(r)) return(p("Run with x=[0.5,1]ᵀ to verify."))
    
    x_test <- c(0.5, 1.0)
    W1 <- r$W1; b1 <- r$b1; W2 <- r$W2; b2 <- r$b2; w3 <- r$w3; b3 <- r$b3
    cp <- r$cp
    
    res_orig <- forward_q2(x_test, W1, b1, W2, b2, w3, b3)
    y_perc   <- sum(cp$w * x_test) + cp$b
    
    tagList(
      p(style="font-size:12px;",
        "Original network: ŷ =", tags$b(round(res_orig$y, 6))), 
      p(style="font-size:12px;",
        "Perceptron: ŷ =", tags$b(round(y_perc, 6))),
      if(abs(res_orig$y - y_perc) < 1e-9) {
        div(class="alert alert-success", style="font-size:12px;padding:6px;",
            "✅ Both match for x = [0.5, 1]ᵀ")
      } else {
        div(class="alert alert-danger", style="font-size:12px;padding:6px;",
            "⚠️ Mismatch detected")
      }
    )
  })
}

# ---- JS for rotation ----
ui$children[[1]]$children[[1]] <- tagAppendChild(
  ui$children[[1]]$children[[1]],
  tags$script(HTML("
    var rotateTimer = null;
    var angle = 0;
    function doRotate() {
      angle += 0.5;
      var rad = angle * Math.PI / 180;
      var plotEl = document.getElementById('plot3d');
      if (plotEl && plotEl._fullLayout) {
        Plotly.relayout(plotEl, {
          'scene.camera.eye': { x: 1.8*Math.cos(rad), y: 1.8*Math.sin(rad), z: 0.8 }
        });
      }
    }
    $(document).on('click','#btn_rotate',function(){
      if(!rotateTimer){ rotateTimer=setInterval(doRotate,30); }
    });
    $(document).on('click','#btn_pause',function(){
      clearInterval(rotateTimer); rotateTimer=null;
    });
  "))
)

shinyApp(ui, server)